import argparse
import json


def get_dice_configs(args_list=None):
    """Parse the DICE/KGE training configuration used by this repository."""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="KGs/Family",
        help="Path to a dataset directory containing train/valid/test triples.",
    )
    parser.add_argument(
        "--sparql_endpoint", type=str, default=None
    )
    parser.add_argument("--path_single_kg", type=str, default=None)
    parser.add_argument("--path_to_store_single_run", type=str, default=None)
    parser.add_argument("--storage_path", type=str, default=None)
    parser.add_argument("--full_storage_path", type=str, default=None)
    parser.add_argument("--save_embeddings_as_csv", action="store_true")
    parser.add_argument(
        "--backend",
        type=str,
        default="pandas",
        choices=["pandas", "polars", "rdflib"],
    )
    parser.add_argument("--separator", type=str, default=r"\s+")

    parser.add_argument(
        "--model",
        type=str,
        default="Keci",
        choices=[
            "ComplEx",
            "Keci",
            "CKeci",
            "ConEx",
            "AConEx",
            "ConvQ",
            "AConvQ",
            "ConvO",
            "AConvO",
            "QMult",
            "OMult",
            "Shallom",
            "DistMult",
            "TransE",
            "DualE",
            "BytE",
            "Pykeen_MuRE",
            "Pykeen_QuatE",
            "Pykeen_DistMult",
            "Pykeen_BoxE",
            "Pykeen_CP",
            "Pykeen_HolE",
            "Pykeen_ProjE",
            "Pykeen_RotatE",
            "Pykeen_TransE",
            "Pykeen_TransF",
            "Pykeen_TransH",
            "Pykeen_TransR",
            "Pykeen_TuckER",
            "Pykeen_ComplEx",
            "LFMult",
            "DeCaL",
        ],
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="Adam",
        choices=["Adam", "AdamW", "SGD", "NAdam", "Adagrad", "ASGD", "Adopt"],
    )
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--callbacks", type=json.loads, default={})
    parser.add_argument(
        "--trainer",
        type=str,
        default="PL",
        choices=["torchCPUTrainer", "PL", "torchDDP", "TP"],
    )
    parser.add_argument(
        "--scoring_technique",
        default="KvsAll",
        choices=["AllvsAll", "KvsAll", "1vsAll", "NegSample", "1vsSample", "KvsSample"],
    )
    parser.add_argument("--neg_ratio", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--input_dropout_rate", type=float, default=0.0)
    parser.add_argument("--hidden_dropout_rate", type=float, default=0.0)
    parser.add_argument("--feature_map_dropout_rate", type=float, default=0.0)
    parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        choices=["LayerNorm", "BatchNorm1d", None],
    )
    parser.add_argument(
        "--init_param",
        type=str,
        default=None,
        choices=["xavier_normal", None],
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=0)
    parser.add_argument("--label_smoothing_rate", type=float, default=0.0)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--num_of_output_channels", type=int, default=2)
    parser.add_argument("--num_core", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--use_manual_training", action="store_true")
    parser.add_argument("--p", type=int, default=0)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--pykeen_model_kwargs", type=json.loads, default={})

    parser.add_argument("--num_folds_for_cv", type=int, default=0)
    parser.add_argument(
        "--eval_model",
        type=str,
        default="train_val_test",
        choices=["None", "train", "train_val", "train_val_test", "test"],
    )
    parser.add_argument("--save_model_at_every_epoch", type=int, default=None)
    parser.add_argument("--continual_learning", type=str, default=None)
    parser.add_argument("--sample_triples_ratio", type=float, default=None)
    parser.add_argument("--read_only_few", type=int, default=None)
    parser.add_argument("--add_noise_rate", type=float, default=0.0)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--byte_pair_encoding", action="store_true")
    parser.add_argument("--adaptive_swa", action="store_true")
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--auto_batch_finding", action="store_true")
    parser.add_argument("--degree", type=int, default=0)
    parser.add_argument("--save_experiment", action="store_true", default=True)
    parser.add_argument("--apply_reciprical_or_noise", action="store_true", default=True)
    parser.add_argument("--log_validation", action="store_true", default=False)
    parser.add_argument("--test_runs", action="store_true", default=False)
    parser.add_argument("--train_all_triples", action="store_true", default=False)
    parser.add_argument("--train_text", action="store_true", default=False)
    parser.add_argument("--early_stopping", action="store_true", default=False)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--adaptive_lr", type=json.loads, default={})
    parser.add_argument("--swa_start_epoch", type=int, default=None)
    parser.add_argument("--eval_every_n_epochs", type=int, default=0)
    parser.add_argument("--save_every_n_epochs", action="store_true")
    parser.add_argument("--eval_at_epochs", type=int, nargs="+", default=None)
    parser.add_argument(
        "--n_epochs_eval_model",
        type=str,
        default="val_test",
        choices=[
            "None",
            "train",
            "train_val",
            "train_val_test",
            "val_test",
            "val",
            "train_test",
            "test",
        ],
    )

    return parser.parse_args(args_list)


def get_default_configs(args_list=None):
    """Parse the local KGEntText configuration for this repository."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train_text", action="store_true", default=False)
    parser.add_argument("--dataset_dir", type=str, default="KGs/FB15k-237")
    parser.add_argument("--pretrained_kge_path", type=str, default=None)
    parser.add_argument("--text_file_name", type=str, default="text_literals.txt")
    parser.add_argument("--full_storage_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--text_model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--model_torch_dtype", type=str, default="auto")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_core", type=int, default=0)
    parser.add_argument("--pad_token_id", type=int, default=0)
    args, _ = parser.parse_known_args(args_list)
    return args
