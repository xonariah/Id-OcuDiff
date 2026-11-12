import json

from PIL import Image
import torch

from .ensemble import load_ensemble, map_feature_timesteps
from .features import extract_features_and_image_at_resolution
from .generation_helpers import load_arc2eyes_components, prompt_pair
from .palette import SEGMENTATION_RGB, tab10_label_image
from .segmentation import segment_feature_map


def generate(settings, sample, with_glasses, image_seed, denoise_seed, device_name="auto"):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    condition = "glasses" if with_glasses else "no_glasses"
    prefix = f"{sample['subject_prefix']}_{int(image_seed)}_{condition}"
    paths = {
        "image": settings.output_dir / f"{prefix}.png",
        "mask": settings.output_dir / f"{prefix}_mask.png",
        "labels": settings.output_dir / f"{prefix}_labels.png",
        "overlay": settings.output_dir / f"{prefix}_overlay.png",
        "metadata": settings.output_dir / f"{prefix}_metadata.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Izhodne datoteke že obstajajo; izberi drugo --seed ali --output-dir:\n" + "\n".join(existing)
        )
    print("Naprava:", device)
    print("Generator:", settings.generator_dir)
    print("Ansambel:", settings.ensemble_manifest_path)
    print("Generiranje:", settings.sample_inference_steps, "korakov | CFG", settings.sample_guidance_scale)
    with torch.no_grad():
        components = load_arc2eyes_components(device, settings)
        ensemble = load_ensemble(settings.ensemble_manifest_path, device)
        blocks, timestep_indices = map_feature_timesteps(ensemble, components, device, settings)
        feature_map, image, initial_latents = extract_features_and_image_at_resolution(
            sample=sample,
            components=components,
            device=device,
            settings=settings,
            feature_hook_blocks=blocks,
            feature_timestep_indices=timestep_indices,
            with_glasses=with_glasses,
            image_seed=image_seed,
            denoise_seed=denoise_seed,
            channel_indices=ensemble["channel_indices"],
            feature_width=settings.width,
            feature_height=settings.height,
        )
        label_mask = segment_feature_map(
            feature_map, ensemble, device, width=image.width, height=image.height,
        )
        color_mask = tab10_label_image(label_mask)
        overlay = Image.blend(image.convert("RGB"), color_mask, alpha=0.50)
    prompt, negative = prompt_pair(with_glasses)
    metadata = {
        "source": sample,
        "with_glasses": bool(with_glasses),
        "image_seed": int(image_seed),
        "denoise_seed": int(denoise_seed),
        "prompt": prompt,
        "negative_prompt": negative,
        "width": int(image.width),
        "height": int(image.height),
        "num_inference_steps": int(settings.sample_inference_steps),
        "guidance_scale": float(settings.sample_guidance_scale),
        "generator_dir": str(settings.generator_dir),
        "base_model_path": str(settings.base_model_path),
        "ensemble_manifest_path": str(settings.ensemble_manifest_path),
        "ensemble_num_models": len(ensemble["models"]),
        "feature_hook_blocks": blocks,
        "feature_timestep_indices": timestep_indices,
        "segmentation_rgb": SEGMENTATION_RGB.tolist(),
        "class_names": {"0": "ozadje", "1": "beločnica", "2": "šarenica", "3": "zenica"},
        "files": {key: str(path) for key, path in paths.items()},
    }
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    image.save(paths["image"])
    color_mask.save(paths["mask"])
    Image.fromarray(label_mask, mode="L").save(paths["labels"])
    overlay.save(paths["overlay"])
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del initial_latents
    return paths
