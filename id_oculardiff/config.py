from typing import Optional


class cfg:
    train_data_dir: str = "./data/train"
    glasses_list_path: str = "./data/glasses_manual.txt"


    train_ocala: bool = True

    output_dir: str = "./models/ID-OcuDiff"
    logging_dir: str = "logs"
    tensorboard_dir: str = "tb_logs"

    pretrained_model_name_or_path: str = (
        "./models/stable-diffusion-2-1"
    )


    resolution: int = 512


    train_batch_size: int = 1
    num_train_epochs: int = 50
    gradient_accumulation_steps: int = 4


    unet_learning_rate: float = 5e-7
    text_encoder_learning_rate: float = 1e-6
    id_adapter_learning_rate: float = 1e-4

    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    seed: int = 42
    mixed_precision: str = "no"
    dataloader_num_workers: int = 2


    prediction_type: str = "v_prediction"
    max_train_steps: Optional[int] = None


    prompt: str = (
        "greyscale, highly detailed eye photo of a id id person "
        "with eyelashes, sharp focus, intricate iris, realistic, photorealistic"
    )
    prompt_glasses: str = (
        "greyscale, highly detailed eye photo of a id id person with glasses "
        "with eyelashes, sharp focus, intricate iris, realistic, photorealistic"
    )
    negative_prompt: str = "low quality, blurry, bad anatomy, deformed, closed eye, glasses"


    sample_every_n_epochs: int = 0
    sample_grid_subjects: int = 4
    sample_grid_variations: int = 4
    sample_inference_steps: int = 70
    sample_guidance_scale: float = 7.0
    sample_seed: int = 42


    save_model_epochs: list[int] = [2, 40, 45]


    id_token_dim: int = 1024
    id_num_tokens: int = 2


    normalize_id: bool = True
