import os

import torch
from accelerate import Accelerator

from .config import cfg


def save_model_checkpoint(
    accelerator: Accelerator,
    unet,
    text_encoder,
    tokenizer,
    id_adapter,
    epoch_num: int,
    global_step: int,
):
    """
    Save an inference-ready checkpoint for a specific epoch:
      checkpoints/epoch_040/
        - unet/
        - text_encoder/
        - tokenizer files
        - id_adapter.pth
        - meta.txt
    """
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", f"epoch_{epoch_num:03d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    accelerator.unwrap_model(unet).save_pretrained(os.path.join(ckpt_dir, "unet"))
    accelerator.unwrap_model(text_encoder).save_pretrained(os.path.join(ckpt_dir, "text_encoder"))
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(
        accelerator.unwrap_model(id_adapter).state_dict(),
        os.path.join(ckpt_dir, "id_adapter.pth"),
    )

    with open(os.path.join(ckpt_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"epoch={epoch_num}\n")
        f.write(f"global_step={global_step}\n")
