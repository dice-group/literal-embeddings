import argparse

def parse_args():
    """Parse command line arguments for model training"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Family")
    parser.add_argument("--model", type=str, default="conve")
    parser.add_argument("--combined_training", action="store_true", help="Use combined training with Literal Embedding model")
    parser.add_argument("--lit_norm", type=str, default="z-norm", help="Normalization type for literal embeddings")
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)  # Reasonable default
    parser.add_argument("--lr", type=float, default=0.001)  # Common learning rate
    parser.add_argument("--lr_decay", type=float, default=1, help="Exponential learning rate decay factor per epoch")
    parser.add_argument("--dr", type=float, default=1.0)
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension for entities, relations, and embedding shape")  # Standard size
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--input_dropout", type=float, default=0.1)  # Basic dropout
    parser.add_argument("--hidden_dropout1", type=float, default=0.3)
    parser.add_argument("--hidden_dropout2", type=float, default=0.3)  # Balanced dropout
    parser.add_argument("--feature_map_dropout", type=float, default=0.2)
    parser.add_argument("--hidden_size", type=float, default=9728)
    parser.add_argument("--label_smoothing", type=float, default=0.0)  # No smoothing by default
    parser.add_argument('--use_bias', action='store_true')
    parser.add_argument("--input", type=str, default="data")
    parser.add_argument("--output", type=str, default="out")
    parser.add_argument("--swa", action="store_true", help="Stochastic weight averaging")
    parser.add_argument("--adaptive_swa", action="store_true", help="Adaptive stochastic weight averaging")
    parser.add_argument("--exp_dir", type=str, default=None, help="Experiment directory to save results")
    parser.add_argument("--dynamic_weighting", action="store_true", help="Use dynamic weighting for training")
    parser.add_argument("--w1", type=float, default=0.95, help="Weight for entity loss in combined training")
    parser.add_argument("--w2", type=float, default=0.05, help="Weight for literal loss in combined training")
    parser.add_argument('--eval_every_n_epochs', type=int, default=0,
                        help='Evaluate model every n epochs. If 0, no evaluation is applied.')
    parser.add_argument('--save_every_n_epochs', action='store_true',
                        help='Save model every n epochs. If True, save model at every epoch.')
    parser.add_argument('--eval_at_epochs',type=int,nargs='+', default=None,
        help="List of epoch numbers at which to evaluate the model (e.g., 1 5 10).")
    parser.add_argument("--n_epochs_eval_model", type=str, default="test",
                        choices=["None", "train", "train_val", "train_val_test", "val_test", "val", "train_test","test"],
                        help='Evaluating link prediction performance on data splits while performing periodic evaluation.')
    
    # Early stopping configuration
    parser.add_argument("--early_stopping", action="store_true", help="Use early stopping based on validation MRR")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--val_check_interval", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum change in validation metric")
    parser.add_argument("--num_core", type=int, default=4, help="Number of CPU cores to use")

    args = parser.parse_args()
    
    # Set model-specific epoch limits based on experimental setup
    if args.model.lower() == "conve":
        if args.num_iterations == 100:  # Only override if using default
            args.num_iterations = 1000  # ConvE needs more epochs as per original paper
    
    # Set unified embedding dimensions
    args.edim = args.embedding_dim
    args.rdim = args.embedding_dim
    
    # Dynamic embedding shape based on embedding dimension
    if args.embedding_dim == 128:
        args.embedding_shape1 = 8  # 8x16 for 128-dim
    elif args.embedding_dim == 200:
        args.embedding_shape1 = 10  # 10x20 for 200-dim
    else:
        # General case: try to make it as square as possible
        import math
        args.embedding_shape1 = int(math.sqrt(args.embedding_dim))
        if args.embedding_dim % args.embedding_shape1 != 0:
            args.embedding_shape1 = args.embedding_dim // 16  # Fallback
    
    # Calculate the correct hidden_size for ConvE based on the conv output
    emb_dim2 = args.embedding_dim // args.embedding_shape1
    conv_out_h = 2 * args.embedding_shape1 - 2  
    conv_out_w = emb_dim2 - 2  
    args.hidden_size = 32 * conv_out_h * conv_out_w
    
    return args