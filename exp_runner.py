from src.config import get_default_arguments
from main import main
args = get_default_arguments()
exp_models = ['TransE', 'DistMult', 'Keci', 'ComplEx', 'OMult', 'QMult', 'DeCaL']

dataset_name = 'FB15k-237'
args.dataset_dir = f'KGs/{dataset_name}'
args.lr = 0.05
args.embedding_dim = 128
args.num_epochs = 256
args.save_experiment = True
args.combined_training = False
args.literal_training = False
for model_name in exp_models:
    args.learning_rate = args.lr
    if args.combined_training:
        args.full_storage_path = f'Experiments/{dataset_name}_{args.embedding_dim}_combined'
    else:
        args.full_storage_path = f'Experiments/{dataset_name}_{args.embedding_dim}'
    args.model = model_name
    main(args)
    print(f"Experiment for {model_name}+{args.embedding_dim} completed and stored at {args.full_storage_path}")
