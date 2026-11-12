import math
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer
from utils.CLIPTextModelWrapper import CLIPTextModelWrapper as CLIPTextModel

from .checkpoints import save_model_checkpoint
from .config import cfg
from .data import EyeDataset2Tokens, collate_fn
from .id_tokens import (
    IDTokenAdapter,
    find_id_positions_in_batch_input_ids,
    inject_id_tokens_no_inplace,
)
from .sampling import generate_sample_grid


def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=cfg.output_dir,
            logging_dir=os.path.join(cfg.output_dir, cfg.logging_dir),
        ),
    )

    writer = None
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(cfg.output_dir, cfg.tensorboard_dir))

    if cfg.seed is not None:
        set_seed(cfg.seed)


    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder")


    noise_scheduler.config.prediction_type = cfg.prediction_type


    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32)


    unet.train()
    text_encoder.train()


    text_encoder.text_model.embeddings.token_embedding.requires_grad_(False)
    if hasattr(text_encoder.text_model.embeddings, "position_embedding"):
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)


    id_adapter = IDTokenAdapter(dim=cfg.id_token_dim, normalize=cfg.normalize_id)

    try:
        id_adapter.init_scale_from_token_embedding(text_encoder.text_model.embeddings.token_embedding)
    except Exception:
        pass


    train_ds = EyeDataset2Tokens(cfg.train_data_dir, cfg.glasses_list_path, cfg.resolution, train_ocala=cfg.train_ocala)
    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=False,
        persistent_workers=(cfg.dataloader_num_workers > 0),
        collate_fn=collate_fn,
    )


    def params_where_grad(m: nn.Module):
        return [p for p in m.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": params_where_grad(unet), "lr": cfg.unet_learning_rate},
            {"params": params_where_grad(text_encoder), "lr": cfg.text_encoder_learning_rate},
            {"params": params_where_grad(id_adapter), "lr": cfg.id_adapter_learning_rate},
        ],
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    steps_per_epoch = math.ceil(len(train_dl) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.max_train_steps or (cfg.num_train_epochs * steps_per_epoch)

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )


    unet, text_encoder, id_adapter, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        unet, text_encoder, id_adapter, optimizer, train_dl, lr_scheduler
    )

    save_model_epochs_set = set(cfg.save_model_epochs)

    global_step = 0
    progress = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(cfg.num_train_epochs):
        for batch in train_dl:
            with accelerator.accumulate(unet, text_encoder, id_adapter):
                imgs = batch["images"].to(dtype=torch.float32)
                bsz = imgs.size(0)


                latents = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor


                noise = torch.randn_like(latents)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, t)


                id_tokens = batch["id_tokens"].to(dtype=torch.float32)
                id_tokens = id_adapter(id_tokens)


                has_glasses = batch["has_glasses"]
                prompt_texts = [
                    cfg.prompt_glasses if (cfg.train_ocala and bool(g.item())) else cfg.prompt
                    for g in has_glasses
                ]
                batch_input_ids = tokenizer(
                    prompt_texts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)

                id_positions = find_id_positions_in_batch_input_ids(
                    tokenizer,
                    batch_input_ids,
                    expected=cfg.id_num_tokens,
                )


                tok_emb = text_encoder.text_model.embeddings.token_embedding(batch_input_ids).to(torch.float32)


                tok_emb = inject_id_tokens_no_inplace(tok_emb, id_positions, id_tokens)


                encoder_states = text_encoder(
                    input_ids=batch_input_ids,
                    input_token_embs=tok_emb,
                    return_dict=False,
                )[0]


                if cfg.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, t)
                else:
                    target = noise


                pred = unet(noisy_latents, t, encoder_states, return_dict=False)[0]
                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(unet.parameters()) + list(text_encoder.parameters()) + list(id_adapter.parameters()),
                        cfg.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress.update(1)
                global_step += 1

                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)

            if global_step >= max_train_steps:
                break


        if cfg.sample_every_n_epochs > 0 and (epoch + 1) % cfg.sample_every_n_epochs == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unet_eval = accelerator.unwrap_model(unet).eval()
                te_eval = accelerator.unwrap_model(text_encoder).eval()
                ida_eval = accelerator.unwrap_model(id_adapter).eval()

                generate_sample_grid(
                    unet=unet_eval,
                    vae=vae,
                    text_encoder=te_eval,
                    tokenizer=tokenizer,
                    id_adapter=ida_eval,
                    device=accelerator.device,
                    step_name=f"epoch{epoch+1}",
                    resolution=cfg.resolution,
                    num_subjects=cfg.sample_grid_subjects,
                    num_variations=cfg.sample_grid_variations,
                    inference_steps=cfg.sample_inference_steps,
                    guidance_scale=cfg.sample_guidance_scale,
                    seed=cfg.sample_seed,
                    prompt_text=cfg.prompt,
                    negative_prompt=cfg.negative_prompt,
                )

                unet_eval.train()
                te_eval.train()
                ida_eval.train()


        current_epoch = epoch + 1
        if current_epoch in save_model_epochs_set:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_model_checkpoint(
                    accelerator=accelerator,
                    unet=unet,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    id_adapter=id_adapter,
                    epoch_num=current_epoch,
                    global_step=global_step,
                )


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        accelerator.unwrap_model(unet).save_pretrained(os.path.join(cfg.output_dir, "unet"))
        accelerator.unwrap_model(text_encoder).save_pretrained(os.path.join(cfg.output_dir, "text_encoder"))
        tokenizer.save_pretrained(cfg.output_dir)
        torch.save(accelerator.unwrap_model(id_adapter).state_dict(), os.path.join(cfg.output_dir, "id_adapter.pth"))

    accelerator.end_training()
