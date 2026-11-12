from dataclasses import dataclass
from pathlib import Path
import os
import re


@dataclass
class GenerationSettings:
    generator_dir: Path = Path("./models/ID-OcuDiff")
    base_model_path: Path = Path("./models/stable-diffusion-2-1")
    ensemble_manifest_path: Path = Path("./models/segmentation/pixel_mlp_ensemble_10/ensemble_manifest.pt")
    raw_dir: Path = Path("./data/test")
    glasses_list_path: Path = Path("./data/glasses_manual.txt")
    output_dir: Path = Path("./outputs/generated")
    width: int = 640
    height: int = 480
    sample_inference_steps: int = 70
    sample_guidance_scale: float = 7.0
    id_num_tokens: int = 2
    id_token_dim: int = 1024
    normalize_id: bool = True


PROMPT_TEXT = (
    "greyscale, highly detailed eye photo of a id id person "
    "with eyelashes, sharp focus, intricate iris, realistic, photorealistic, eyebrow,"
    "photorealistic, realistic photograph, raw photo, depth of field, film grain, textured skin, sharp"
)
NEGATIVE_PROMPT = (
    "low quality, blurry, bad anatomy, deformed, closed eye, glasses, "
    "waxy skin, plastic skin, airbrushed skin, oversmoothed, cgi, render, illustration,"
    "ugly skin, celebrity, Ugly, low, deformed, smooth, blurry"
)


def normalize_filename_key(path_or_name: str) -> str:
    return str(path_or_name).replace("\\", "/").rsplit("/", 1)[-1].strip().upper()


def load_glasses_set(glasses_file: Path) -> set:
    if glasses_file.exists():
        with open(glasses_file, "r", encoding="utf-8") as f:
            return {
                normalize_filename_key(line)
                for line in f
                if line.strip()
            }

    print(f"[WARN] glasses list not found: {glasses_file}. Glasses filtering disabled.")
    return set()


def extract_subject_keys(filename: str):
    """
    Primarni format:
      S5XXXLYY ali S5XXXRYY

    Vrne:
      subject_prefix = S5XXXL ali S5XXXR
      subject_code   = S5XXXL ali S5XXXR
      subject_id     = XXX

    L/R ostane del identitete, enako kot v tvoji kodi.
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    m = re.match(r"^(S\d(\d{3})([LR]))(\d+)$", base, re.IGNORECASE)
    if m:
        subject_prefix = m.group(1).upper()
        subject_code = subject_prefix
        subject_id = m.group(2)
        return subject_prefix, subject_code, subject_id


    m = re.match(r"^([A-Z]?\d+[LR])(\d+)$", base, re.IGNORECASE)
    if m:
        subject_prefix = m.group(1).upper()
        digits = re.findall(r"\d+", subject_prefix)
        if digits:
            subject_id = digits[-1][-3:].zfill(3)
            subject_code = subject_prefix
            return subject_prefix, subject_code, subject_id

    return None, None, None


def collect_train_subjects(
    train_root: Path,
    glasses_file: Path = None,
    exclude_glasses: bool = True,
):
    """
    Rekurzivno prebere RAW_DIR in poišče slike, ki imajo pripadajoča:
      *_1.pt
      *_2.pt

    Grupira po subject_code = S5XXXL ali S5XXXR.
    """
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    glasses_set = load_glasses_set(glasses_file) if (exclude_glasses and glasses_file is not None) else set()

    by_subject_code = {}

    for current_root, _, files in os.walk(train_root):
        for fn in sorted(files):
            fn_lower = fn.lower()

            if not fn_lower.endswith(valid_ext):
                continue
            if "_seg" in fn_lower or "_mask" in fn_lower:
                continue


            if exclude_glasses and normalize_filename_key(fn) in glasses_set:
                continue

            image_path = Path(current_root) / fn
            base_path = image_path.with_suffix("")
            pt1 = Path(str(base_path) + "_1.pt")
            pt2 = Path(str(base_path) + "_2.pt")

            if not (pt1.exists() and pt2.exists()):
                continue

            subject_prefix, subject_code, subject_id = extract_subject_keys(fn)
            if subject_prefix is None or subject_code is None or subject_id is None:
                continue

            item = {
                "image_path": str(image_path),
                "pt1_path": str(pt1),
                "pt2_path": str(pt2),
                "subject_prefix": subject_prefix,
                "subject_code": subject_code,
                "subject_id": subject_id,
                "filename": fn,
            }

            by_subject_code.setdefault(subject_code, []).append(item)

    if not by_subject_code:
        raise RuntimeError(
            f"V RAW_DIR nisem našla ustreznih vzorcev: {train_root}\n"
            "Pričakujem slike + pripadajoča *_1.pt in *_2.pt vektorja."
        )

    return by_subject_code


def remove_glasses_from_negative(text: str) -> str:
    return ", ".join(
        part.strip() for part in text.split(",")
        if part.strip() and not re.search(r"\bglasses\b", part, flags=re.IGNORECASE)
    )


def prompt_pair(with_glasses: bool):
    if with_glasses:
        return PROMPT_TEXT.rstrip(" ,") + ", with glasses", remove_glasses_from_negative(NEGATIVE_PROMPT)
    return PROMPT_TEXT, NEGATIVE_PROMPT


def select_sample(settings, image_path, id1, id2, with_glasses, rng):
    if (id1 is None) != (id2 is None):
        raise ValueError("Podaj oba argumenta --id1 in --id2.")
    if image_path is not None or id1 is not None:
        image_path = Path(image_path) if image_path is not None else None
        if image_path is not None and not image_path.is_file():
            raise FileNotFoundError(f"Ne najdem vhodne slike: {image_path}")
        if id1 is None:
            base_path = image_path.with_suffix("")
            id1 = Path(str(base_path) + "_1.pt")
            id2 = Path(str(base_path) + "_2.pt")
        id1, id2 = Path(id1), Path(id2)
        for path in [id1, id2]:
            if not path.is_file():
                raise FileNotFoundError(f"Ne najdem ID-vektorja: {path}")
        filename = image_path.name if image_path is not None else id1.name
        prefix, code, subject_id = extract_subject_keys(filename)
        return {
            "image_path": str(image_path) if image_path is not None else None,
            "pt1_path": str(id1),
            "pt2_path": str(id2),
            "subject_prefix": prefix or "identity",
            "subject_code": code,
            "subject_id": subject_id,
            "filename": filename,
        }
    if not settings.raw_dir.is_dir():
        raise FileNotFoundError(f"Ne najdem mape s slikami in ID-vektorji: {settings.raw_dir}")
    if not settings.glasses_list_path.is_file():
        raise FileNotFoundError(
            f"Za samodejni izbor po pogoju očal potrebujem {settings.glasses_list_path}. "
            "Lahko podaš tudi --image ali oba --id1 in --id2."
        )
    by_subject = collect_train_subjects(settings.raw_dir, exclude_glasses=False)
    glasses_names = load_glasses_set(settings.glasses_list_path)
    candidates = {}
    for code, samples in by_subject.items():
        matching = [
            sample for sample in samples
            if (normalize_filename_key(sample["filename"]) in glasses_names) == with_glasses
        ]
        if matching:
            candidates[code] = matching
    if not candidates:
        condition = "z očali" if with_glasses else "brez očal"
        raise RuntimeError(f"V {settings.raw_dir} ni vzorcev {condition} z obema ID-vektorjema.")
    code = rng.choice(sorted(candidates))
    return rng.choice(candidates[code])


def validate_generation_assets(settings):
    required = [
        settings.generator_dir / "unet",
        settings.generator_dir / "text_encoder",
        settings.generator_dir / "id_adapter.pth",
        settings.base_model_path / "vae",
        settings.base_model_path / "scheduler",
        settings.ensemble_manifest_path,
        Path("./utils/CLIPTextModelWrapper.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Manjkajo potrebne datoteke ali mape:\n" + "\n".join(missing))


def load_tokenizer(settings):
    from transformers import CLIPTokenizer

    try:
        return CLIPTokenizer.from_pretrained(str(settings.generator_dir))
    except Exception:
        return CLIPTokenizer.from_pretrained(str(settings.base_model_path), subfolder="tokenizer")


def build_scheduler(settings):
    from diffusers import DPMSolverMultistepScheduler

    base_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        str(settings.base_model_path),
        subfolder="scheduler",
    )
    try:
        scheduler = DPMSolverMultistepScheduler.from_config(
            base_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            solver_order=2,
        )
    except TypeError:
        scheduler = base_scheduler
    return scheduler


def load_arc2eyes_components(device, settings):
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from utils.CLIPTextModelWrapper import CLIPTextModelWrapper as CLIPTextModel
    from .id_tokens import IDTokenAdapter

    vae = AutoencoderKL.from_pretrained(
        str(settings.base_model_path), subfolder="vae",
    ).to(device, dtype=torch.float32)
    scheduler = build_scheduler(settings)
    tokenizer = load_tokenizer(settings)
    text_encoder = CLIPTextModel.from_pretrained(
        str(settings.generator_dir / "text_encoder")
    ).to(device, dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(
        str(settings.generator_dir / "unet")
    ).to(device, dtype=torch.float32)
    id_adapter_path = settings.generator_dir / "id_adapter.pth"
    id_adapter = IDTokenAdapter(
        dim=settings.id_token_dim, normalize=settings.normalize_id,
    ).to(device, dtype=torch.float32)
    id_adapter.load_state_dict(torch.load(str(id_adapter_path), map_location=device))
    for model in [vae, text_encoder, unet, id_adapter]:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return {
        "vae": vae,
        "scheduler": scheduler,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "unet": unet,
        "id_adapter": id_adapter,
    }
