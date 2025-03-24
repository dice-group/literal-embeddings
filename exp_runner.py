from src.config import get_default_arguments
from main import main, train_with_kge
import os

args = get_default_arguments()
exp_models = ["TransE", "DistMult", "Keci", "ComplEx", "OMult", "QMult", "DeCaL"]
# exp_models = ["TransE"]
exp_dir = "Experiments/Family_128"
args.literal_training = False

if args.literal_training:
    for model_name in exp_models:
        sub_path = os.path.join(exp_dir, model_name)
        args.num_literal_runs = 3
        args.pretrained_kge_path = sub_path
        res = train_with_kge(args)

else:
    dataset_name = "FB15k-237"
    args.dataset_dir = f"KGs/{dataset_name}"
    args.lr = 0.05
    args.embedding_dim = 128
    args.num_epochs = 256
    args.save_experiment = True
    args.combined_training = True

    for model_name in exp_models:
        args.learning_rate = args.lr
        if args.combined_training:
            args.full_storage_path = (
                f"Experiments/{dataset_name}_{args.embedding_dim}_combined/{model_name}"
            )
        else:
            args.full_storage_path = (
                f"Experiments/{dataset_name}_{args.embedding_dim}/{model_name}"
            )
        args.model = model_name
        main(args)
        print(
            f"Experiment for {model_name}+{args.embedding_dim} completed and stored at {args.full_storage_path}"
        )
