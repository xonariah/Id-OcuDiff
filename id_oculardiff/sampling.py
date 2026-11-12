import os

import torch
from diffusers import DPMSolverMultistepScheduler
from torchvision import utils as tv_utils

from .config import cfg
from .data import EyeDataset2Tokens
from .id_tokens import (
    IDTokenAdapter,
    ensure_1d,
    find_id_positions_in_batch_input_ids,
    inject_id_tokens_no_inplace,
)


@torch.no_grad()
def generate_sample_grid(
    unet,
    vae,
    text_encoder,
    tokenizer,
    id_adapter: IDTokenAdapter,
    device: torch.device,
    step_name: str,
    resolution: int,
    num_subjects: int,
    num_variations: int,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
    prompt_text: str,
    negative_prompt: str,
):
    """
    Save a 4×4 grid:
      - 4 rows: 4 distinct subjects
      - 4 cols: 4 stochastic variations for the same ID
    """

    longer = resolution
    shorter = int(resolution * 3 / 4)
    height, width = (shorter, longer) if longer > shorter else (longer, shorter)

    torch.manual_seed(seed)


    ds = EyeDataset2Tokens(cfg.train_data_dir, cfg.glasses_list_path, resolution, train_ocala=cfg.train_ocala)


    subj_to_first_idx = {}
    for i, sid in enumerate(ds.subject_ids):
        if sid not in subj_to_first_idx:
            subj_to_first_idx[sid] = i
    all_subjects = sorted(subj_to_first_idx.keys(), key=lambda x: (x == -1, x))
    chosen_subjects = all_subjects[:num_subjects]
    chosen_indices = [subj_to_first_idx[sid] for sid in chosen_subjects]


    id_list = []
    glasses_flags = []
    for i in chosen_indices:
        id1 = ensure_1d(torch.load(ds.pt1_paths[i], map_location="cpu"))
        id2 = ensure_1d(torch.load(ds.pt2_paths[i], map_location="cpu"))
        id_tokens = torch.stack([id1, id2], dim=0)
        id_list.append(id_tokens.unsqueeze(0).repeat(num_variations, 1, 1))
        glasses_flags.extend([ds.has_glasses[i]] * num_variations)
    id_batch = torch.cat(id_list, dim=0).to(device)
    n = id_batch.shape[0]


    id_batch = id_adapter(id_batch)


    prompt_texts = [
        cfg.prompt_glasses if (cfg.train_ocala and has_glasses) else prompt_text
        for has_glasses in glasses_flags
    ]
    batch_input_ids = tokenizer(
        prompt_texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    id_positions = find_id_positions_in_batch_input_ids(tokenizer, batch_input_ids, expected=cfg.id_num_tokens)


    tok_emb = text_encoder.text_model.embeddings.token_embedding(batch_input_ids).to(torch.float32)


    tok_emb = inject_id_tokens_no_inplace(tok_emb, id_positions, id_batch)


    cond = text_encoder(input_ids=batch_input_ids, input_token_embs=tok_emb, return_dict=False)[0]


    uncond_ids = tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    uncond = text_encoder(input_ids=uncond_ids.repeat(n, 1), return_dict=False)[0]

    all_emb = torch.cat([uncond, cond], dim=0)


    dpm = DPMSolverMultistepScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    dpm.set_timesteps(inference_steps, device=device)

    latents = torch.randn((n, 4, height // 8, width // 8), device=device, dtype=torch.float32)

    for t in dpm.timesteps:
        inp = torch.cat([latents, latents], dim=0)
        noise_pred = unet(inp, t, encoder_hidden_states=all_emb).sample
        u, c = noise_pred.chunk(2, dim=0)
        guided = u + guidance_scale * (c - u)
        latents = dpm.step(guided, t, latents).prev_sample

    imgs = vae.decode(latents / vae.config.scaling_factor).sample
    imgs = (imgs.clamp(-1, 1) + 1) / 2

    grid = tv_utils.make_grid(imgs, nrow=num_variations)
    os.makedirs(cfg.output_dir, exist_ok=True)
    tv_utils.save_image(grid, os.path.join(cfg.output_dir, f"{step_name}_samples.png"))


    with open(os.path.join(cfg.output_dir, f"{step_name}_subjects.txt"), "w", encoding="utf-8") as f:
        f.write("Row order (top->bottom) subject IDs:\n")
        for sid in chosen_subjects:
            f.write(f"{sid}\n")
