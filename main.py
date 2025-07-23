### Main Entry Point
from src.config import get_default_arguments
from train_kge import train_kge_model
from train_literals import train_literals


def main():
    """Main entry point for the literal embeddings training framework."""
    args = get_default_arguments()
    args.learning_rate = args.lr
    
    if args.literal_training:
        train_literals(args)
    else:
        train_kge_model(args)


if __name__ == "__main__":
    main()
# This script is the main entry point for training and evaluating a KGE model with literal embeddings.