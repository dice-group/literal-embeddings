import argparse

def parse_args():
    """Parse command line arguments for model training"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Family")
    parser.add_argument("--model", type=str, default="conve_literal")
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.95, help="Exponential learning rate decay factor per epoch")
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

class Data:
    """Data loader for knowledge graph triples"""
    def __init__(self, data_dir="data/FB15k-237/", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train", reverse=False):
        """Load triples from file, optionally adding reverse triples"""
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        """Extract unique relations from triples"""
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        """Extract unique entities from triples"""
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities