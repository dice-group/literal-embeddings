from runners.kge_runner import train_kge_model
from runners.runner_KGEntText import train_kgenttext_model
from src.config import get_default_configs, get_dice_configs


def main():
    """Main entry point for KGE and KGEntText training."""
    local_args = get_default_configs()
    if local_args.train_text:
        train_kgenttext_model(local_args)
    else:
        args = get_dice_configs()
        train_kge_model(args)


if __name__ == "__main__":
    main()
