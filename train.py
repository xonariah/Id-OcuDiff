import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Train ID-OcuDiff using id_oculardiff/config.py."
    )
    parser.add_argument("--output-dir", help="Override the model output directory.")
    args = parser.parse_args()

    if args.output_dir is not None:
        from id_oculardiff.config import cfg

        cfg.output_dir = args.output_dir

    from id_oculardiff.training import main as train

    train()


if __name__ == "__main__":
    main()
