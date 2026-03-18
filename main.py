from runners.kge_runner import train_kge_model
from src.config import get_default_arguments


def main():
    """Main entry point for KGE training."""
    args = get_default_arguments()
    args.learning_rate = args.lr
    train_kge_model(args)


if __name__ == "__main__":
    main()
