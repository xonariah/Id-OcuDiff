import argparse
import os
from pathlib import Path
import random

from id_oculardiff.generation_helpers import (
    GenerationSettings,
    select_sample,
    validate_generation_assets,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Generiranje slike ID-OcuDiff in pripadajoče semantične maske.")
    parser.add_argument("-glasses", "--glasses", action="store_true", help="Generiraj sliko z očali.")
    parser.add_argument("--image", type=Path, help="Izvorna slika s pripadajočima *_1.pt in *_2.pt.")
    parser.add_argument("--id1", type=Path, help="Prvi pripravljeni ID-vektor.")
    parser.add_argument("--id2", type=Path, help="Drugi pripravljeni ID-vektor.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/test"))
    parser.add_argument("--glasses-list", type=Path, default=Path("./data/glasses_manual.txt"))
    parser.add_argument("--model-dir", type=Path, default=Path("./models/ID-OcuDiff"))
    parser.add_argument("--base-model", type=Path, default=Path("./models/stable-diffusion-2-1"))
    parser.add_argument(
        "--ensemble-manifest", type=Path,
        default=Path("./models/segmentation/pixel_mlp_ensemble_10/ensemble_manifest.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/generated"))
    parser.add_argument("--seed", type=int, help="Seme za izbor identitete in začetni latent.")
    parser.add_argument("--denoise-seed", type=int, help="Seme za naključnost SDE schedulerja.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = GenerationSettings(
        generator_dir=args.model_dir,
        base_model_path=args.base_model,
        ensemble_manifest_path=args.ensemble_manifest,
        raw_dir=args.data_dir,
        glasses_list_path=args.glasses_list,
        output_dir=args.output_dir,
    )
    image_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2**32 - 1)
    rng = random.Random(image_seed)
    try:
        sample = select_sample(settings, args.image, args.id1, args.id2, args.glasses, rng)
        validate_generation_assets(settings)
        denoise_seed = args.denoise_seed if args.denoise_seed is not None else rng.randrange(0, 2**32 - 1)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from id_oculardiff.generation import generate

        paths = generate(settings, sample, args.glasses, image_seed, denoise_seed, args.device)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Napaka: {error}\n")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
