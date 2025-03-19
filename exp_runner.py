from src.config import get_default_arguments
from main import main
args = get_default_arguments()
exp_models = ['TransE', 'DistMult', 'Keci', 'ComplEx', 'OMult', 'QMult', 'DeCaL']

dataset_name = 'FB15k-237'
args.dataset_dir = f'KGs/{dataset_name}'
args.lr = 0.05
args.embedding_dim = 128
args.num_epochs = 256
dataset_name = 
args.save_experiment = True
for model_name in exp_models:
    if args.combined_training:
        args.full_storage_path = f'Experiments/{dataset_name}_{args.embedding_dim}'
    args.model = model_name
    main(args)
