### Main Entry Point
from runners.kge_runner import train_kge_model
from runners.literal_runner import train_literals
from runners.ff_runner import train_kge_ff
from src.config import get_default_arguments


def main():
    """Main entry point for the literal embeddings training framework."""
    args = get_default_arguments()
    args.learning_rate = args.lr
    
    if args.literal_training:
        train_literals(args)
    elif args.ff_training:
        train_kge_ff(args)
    else:
        train_kge_model(args)


if __name__ == "__main__":
    main()
# This script is the main entry point for training and evaluating a KGE model with literal embeddings.