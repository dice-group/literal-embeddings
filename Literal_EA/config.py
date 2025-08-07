import argparse

def parse_args():
    """Parse command line arguments for model training"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Family")
    parser.add_argument("--model", type=str, default="conve")
    parser.add_argument("--combined_training", action="store_true", help="Use combined training with Literal Embedding model")
    parser.add_argument("--lit_norm", type=str, default="z-norm", help="Normalization type for literal embeddings")
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_decay", type=float, default=1, help="Exponential learning rate decay factor per epoch")
    parser.add_argument("--dr", type=float, default=1.0)
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension for entities, relations, and embedding shape")
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--input_dropout", type=float, default=0.0)
    parser.add_argument("--hidden_dropout1", type=float, default=0.4)
    parser.add_argument("--hidden_dropout2", type=float, default=0.5)
    parser.add_argument("--feature_map_dropout", type=float, default=0.2)
    parser.add_argument("--hidden_size", type=float, default=9728)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument('--use_bias', action='store_true')
    parser.add_argument("--input", type=str, default="data")
    parser.add_argument("--output", type=str, default="out")
    parser.add_argument("--swa", action="store_true", help="Stochastic weight averaging")
    parser.add_argument("--adaptive_swa", action="store_true", help="Adaptive stochastic weight averaging")
    parser.add_argument("--exp_dir", type=str, default=None, help="Experiment directory to save results")
    parser.add_argument("--dynamic_weighting", action="store_true", help="Use dynamic weighting for training")
    parser.add_argument("--w1", type=float, default=0.95, help="Weight for entity loss in combined training")
    parser.add_argument("--w2", type=float, default=0.05, help="Weight for literal loss in combined training")
    
    args = parser.parse_args()
    
    # Set unified embedding dimensions
    args.edim = args.embedding_dim
    args.rdim = args.embedding_dim
    # For ConvE with 128-dim embeddings, use 8 for balanced 16x16 conv input
    args.embedding_shape1 = 8
    
    # Calculate the correct hidden_size for ConvE based on the conv output
    # After reshape: (batch, 1, emb_dim1, emb_dim2) = (batch, 1, 8, 16)
    # After concat: (batch, 1, 16, 16)
    # After 3x3 conv: (batch, 32, 14, 14)
    # After flatten: batch, 32 * 14 * 14 = batch, 6272
    emb_dim2 = args.embedding_dim // args.embedding_shape1
    conv_out_h = 2 * args.embedding_shape1 - 2  # 16 - 2 = 14
    conv_out_w = emb_dim2 - 2  # 16 - 2 = 14
    args.hidden_size = 32 * conv_out_h * conv_out_w  # 32 * 14 * 14 = 6272
    
    return args