import gc
import inspect

import torch
import torch.nn.functional as F
from PIL import Image

from .generation_helpers import prompt_pair
from .id_tokens import ensure_1d, find_id_token_positions, inject_id_tokens_no_inplace


def seed_torch_global(seed: int):
    """
    Nastavi globalni torch RNG.
    To uporabimo kot fallback za schedulerje, ki nimajo generator= argumenta.
    """
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_torch_generator(seed: int, device):
    """
    Generator na isti napravi kot model.
    """
    rng_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.Generator(device=rng_device).manual_seed(int(seed))


def scheduler_step_deterministic(scheduler, guided, t, latents, generator=None):
    """
    Pri sde-dpmsolver++ je lahko v scheduler.step dodatna naključnost.
    Če scheduler podpira generator=..., ga podamo.
    """
    step_signature = inspect.signature(scheduler.step)

    if "generator" in step_signature.parameters:
        return scheduler.step(
            guided,
            t,
            latents,
            generator=generator,
        ).prev_sample

    return scheduler.step(
        guided,
        t,
        latents,
    ).prev_sample


def reset_scheduler_before_generation(scheduler, device, settings):
    """
    set_timesteps običajno resetira notranje stanje schedulerja.
    Tukaj dodamo še eksplicitni reset za varnost.
    """
    scheduler.set_timesteps(settings.sample_inference_steps, device=device)

    if hasattr(scheduler, "model_outputs"):
        try:
            scheduler.model_outputs = [None] * scheduler.config.solver_order
        except Exception:
            pass

    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0

    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None


def get_module_by_path(root_module, path: str):
    """
    Primer:
      get_module_by_path(unet, "up_blocks.2")
    """
    obj = root_module

    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)

    return obj


def first_4d_tensor(output):
    """
    Iz hook outputa poišče prvi 4D tensor [B, C, H, W].
    """
    if isinstance(output, torch.Tensor):
        if output.ndim == 4:
            return output
        return None

    if isinstance(output, (tuple, list)):
        for item in output:
            found = first_4d_tensor(item)
            if found is not None:
                return found

    return None


def load_subject_id_tokens(sample: dict, settings) -> torch.Tensor:
    id1 = ensure_1d(torch.load(sample["pt1_path"], map_location="cpu"))
    id2 = ensure_1d(torch.load(sample["pt2_path"], map_location="cpu"))

    if id1.numel() != settings.id_token_dim or id2.numel() != settings.id_token_dim:
        raise ValueError(
            f"Napačna dimenzija ID tokenov za {sample['filename']}: "
            f"id1={id1.numel()}, id2={id2.numel()}, pričakujem {settings.id_token_dim}."
        )

    return torch.stack([id1, id2], dim=0)


def build_conditioning_for_sample(sample, components, device, settings, with_glasses, batch_size=1):
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    id_adapter = components["id_adapter"]
    prompt, negative = prompt_pair(with_glasses)
    prompt_ids, id_positions = find_id_token_positions(
        tokenizer=tokenizer, prompt=prompt, device=device, expected=settings.id_num_tokens,
    )
    negative_ids = tokenizer(
        [negative], padding="max_length", max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    id_tokens = load_subject_id_tokens(sample, settings).unsqueeze(0)
    id_tokens = id_tokens.repeat(batch_size, 1, 1).to(device, dtype=torch.float32)
    id_tokens = id_adapter(id_tokens)
    batch_input_ids = prompt_ids.repeat(batch_size, 1)
    tok_emb = text_encoder.text_model.embeddings.token_embedding(batch_input_ids)
    tok_emb = tok_emb.to(torch.float32)
    tok_emb = inject_id_tokens_no_inplace(
        tok_emb=tok_emb, positions_per_sample=[id_positions] * batch_size, id_tokens=id_tokens,
    )
    cond = text_encoder(
        input_ids=batch_input_ids, input_token_embs=tok_emb, return_dict=False,
    )[0]
    uncond = text_encoder(
        input_ids=negative_ids.repeat(batch_size, 1), return_dict=False,
    )[0]
    all_emb = torch.cat([uncond, cond], dim=0)
    return all_emb


def extract_features_and_image_at_resolution(
    sample: dict,
    components,
    device,
    settings,
    feature_hook_blocks,
    feature_timestep_indices,
    with_glasses: bool,
    image_seed: int,
    denoise_seed: int,
    channel_indices=None,
    feature_width: int = 640,
    feature_height: int = 480,
):
    """
    Regenerira sliko in med istim denoisingom pobere U-Net feature mape.

    Vrne:
      feature_map: [C, feature_height, feature_width] CPU float16
      pil_img:     PIL image full-res
      initial_latents_cpu_out: začetni latent na CPU
    """
    FEATURE_HOOK_BLOCKS = feature_hook_blocks
    FEATURE_TIMESTEP_INDICES = feature_timestep_indices
    HEIGHT, WIDTH = settings.height, settings.width
    SAMPLE_GUIDANCE_SCALE = settings.sample_guidance_scale
    vae = components["vae"]
    scheduler = components["scheduler"]
    unet = components["unet"]

    batch_size = 1

    all_emb = build_conditioning_for_sample(
        sample, components, device, settings, with_glasses, batch_size=batch_size,
    )

    reset_scheduler_before_generation(scheduler, device, settings)

    latent_shape = (batch_size, 4, HEIGHT // 8, WIDTH // 8)
    latent_generator = make_torch_generator(image_seed, device)

    latents = torch.randn(
        latent_shape,
        generator=latent_generator,
        device=device,
        dtype=torch.float32,
    )
    initial_latents_cpu_out = latents.detach().cpu().clone()

    captured = {}
    current_step = {"idx": None}
    handles = []

    def make_hook(block_name):
        def hook(module, inputs, output):
            step_idx = current_step["idx"]

            if step_idx not in FEATURE_TIMESTEP_INDICES:
                return

            tensor = first_4d_tensor(output)
            if tensor is None:
                return


            if tensor.shape[0] >= 2 * batch_size:
                tensor = tensor[-batch_size:]
            else:
                tensor = tensor[:batch_size]

            captured[(step_idx, block_name)] = tensor.detach().to("cpu", dtype=torch.float32)

        return hook

    for block_name in FEATURE_HOOK_BLOCKS:
        module = get_module_by_path(unet, block_name)
        handles.append(module.register_forward_hook(make_hook(block_name)))

    seed_torch_global(int(denoise_seed))
    denoise_generator = make_torch_generator(denoise_seed, device)

    try:
        for step_idx, t in enumerate(scheduler.timesteps):
            current_step["idx"] = step_idx

            inp = torch.cat([latents, latents], dim=0)

            noise_pred = unet(
                inp,
                t,
                encoder_hidden_states=all_emb,
            ).sample

            noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
            guided = noise_uncond + SAMPLE_GUIDANCE_SCALE * (noise_cond - noise_uncond)

            latents = scheduler_step_deterministic(
                scheduler=scheduler,
                guided=guided,
                t=t,
                latents=latents,
                generator=denoise_generator,
            )

    finally:
        for h in handles:
            h.remove()

    upsampled = []
    missing = []

    for step_idx in FEATURE_TIMESTEP_INDICES:
        for block_name in FEATURE_HOOK_BLOCKS:
            key = (step_idx, block_name)

            if key not in captured:
                missing.append(key)
                continue

            fm = captured[key]
            fm = F.interpolate(
                fm,
                size=(feature_height, feature_width),
                mode="bilinear",
                align_corners=False,
            )
            upsampled.append(fm)

    if missing:
        print("Captured keys:", sorted(captured.keys()))
        raise RuntimeError(f"Manjkajo feature mape za: {missing}")

    feature_map = torch.cat(upsampled, dim=1)[0].contiguous()

    if channel_indices is not None:
        feature_map = feature_map[channel_indices]

    feature_map = feature_map.to(torch.float16).cpu()

    imgs = vae.decode(latents / vae.config.scaling_factor).sample
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    img_np = imgs[0].detach().cpu().permute(1, 2, 0).numpy()
    pil_img = Image.fromarray((img_np * 255).astype("uint8"))

    del captured, upsampled, latents, imgs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return feature_map, pil_img, initial_latents_cpu_out
