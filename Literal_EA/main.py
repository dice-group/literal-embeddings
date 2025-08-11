import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from config import parse_args
from callbacks import ASWA, PeriodicEvalCallback
from litem import LiteralEmbeddings

# Set float32 matmul precision for better performance
torch.set_float32_matmul_precision('medium')

from dataset import KGDataset , LiteralDataset
from model import *
from trainer import LitModel
from evaluate import Evaluator
import json
from pytorch_lightning.callbacks.stochastic_weight_avg import \
    StochasticWeightAveraging as SWA
from pytorch_lightning.callbacks import EarlyStopping

# Map model names to model classes
model_mapping = {
    "tucker": TuckER,
    "tucker_literal": TuckER_Literal,
    "tucker_kbln": TuckER_KBLN,
    "conve": ConvE,
    "conve_literal": ConvE_Literal,
    "conve_kbln": ConvE_KBLN,
    "distmult": DistMult,
    "distmult_literal": DistMult_Literal,
    "distmult_kbln": DistMult_KBLN,
    "complex": ComplEx,
    "complex_literal": ComplEx_Literal,
    "complex_kbln": ComplEx_KBLN
}

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    h, r, t = zip(*batch)
    return (torch.tensor(h), torch.tensor(r), torch.tensor(t))

def build_er_vocab(triples, entity2idx, relation2idx):
    """Build entity-relation vocabulary for filtered evaluation"""
    er_vocab = dict()
    for h, r, t in triples:
        h_idx = entity2idx[h]
        r_idx = relation2idx[r]
        t_idx = entity2idx[t]
        er_vocab.setdefault((h_idx, r_idx), []).append(t_idx)
    return er_vocab

def main(args):
    """Main training function"""
    
    data_dir = f"{args.input}/{args.dataset}/"
    
    # Set experiment directory if not provided by user
    if args.exp_dir is None:
        if args.combined_training:
            args.exp_dir = f"Experiments/Literals_combined/{args.dataset}_{args.model}"
        else:
            args.exp_dir = f"Experiments/Literals/{args.dataset}_{args.model}"
        if args.swa:
            args.exp_dir = args.exp_dir + '/SWA'
        if args.adaptive_swa:
            args.exp_dir = args.exp_dir + "/ASWA"
    
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
        
    # Set random seed for reproducibility
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create datasets
    train_dataset = KGDataset(data_dir=data_dir, split="train", reverse=True)
    val_dataset   = KGDataset(data_dir=data_dir, split="valid", reverse=True)
    test_dataset  = KGDataset(data_dir=data_dir, split="test",  reverse=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=10, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=10, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=1, collate_fn=collate_fn)
    
    # Model configuration
    kwargs = {
        "input_dropout": args.input_dropout,
        "hidden_dropout1": args.hidden_dropout1,
        "hidden_dropout2": args.hidden_dropout2,
        "feature_map_dropout": args.feature_map_dropout,
        "hidden_size": args.hidden_size,
        "use_bias": args.use_bias,
        "embedding_shape1": args.embedding_shape1,
        "dataset": args.dataset,
        "ent2idx": train_dataset.entity2idx,
        "rel2idx": train_dataset.relation2idx,
        "device": "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    }
    
    # Build evaluation vocabulary
    all_triples = (train_dataset.triples + val_dataset.triples + test_dataset.triples)
    er_vocab = build_er_vocab(all_triples, train_dataset.entity2idx, train_dataset.relation2idx) 

    # Initialize evaluator
    evaluator = Evaluator(
        er_vocab=er_vocab,  
        train_datalaoder=train_loader,
        test_dataloader=test_loader,
        val_dataloader=val_loader)

    
    if args.combined_training:
        print("Using combined training with Literal Embedding model")
        literal_dataset = LiteralDataset(
        dataset_dir=data_dir,
        ent_idx=train_dataset.entity2idx,
        normalization_type=args.lit_norm
    )
    
        litem_model = LiteralEmbeddings(
        num_of_data_properties=literal_dataset.num_data_properties,
        embedding_dims=args.embedding_dim,
        dropout=0.15, freeze_entity_embeddings = False)
    else:
        litem_model = None
        literal_dataset = None
    # Initialize model
    lit_model = LitModel(
        args, train_dataset, kwargs, model_mapping=model_mapping,
        er_vocab=er_vocab, evaluator=evaluator, 
        literal_model=litem_model, literal_dataset=literal_dataset
    )
    
    # Setup callbacks
    callbacks  = []
    
    # Add early stopping callback
    if hasattr(args, 'early_stopping') and args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_mrr",
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=True,
            mode="max"  # We want to maximize MRR
        )
        callbacks.append(early_stop_callback)
        print(f"Using early stopping: monitoring val_mrr with patience={args.patience}")
    
    if args.swa:
        callbacks.append(SWA(swa_epoch_start=1, swa_lrs=args.lr))
        print("Using Stochastic Weight Averaging (SWA)")
    elif args.adaptive_swa:
        callbacks.append(ASWA(path=args.exp_dir, num_epochs=args.num_iterations))
        print("Using Adaptive Stochastic Weight Averaging (ASWA)") 
    else:
        print("Not using weight averaging")
    
    if args.eval_every_n_epochs > 0 or args.eval_at_epochs is not None:
        callbacks.append(PeriodicEvalCallback(experiment_path=args.exp_dir, max_epochs=args.num_iterations,
                        eval_every_n_epoch=args.eval_every_n_epochs, eval_at_epochs=args.eval_at_epochs,
                        save_model_every_n_epoch=args.save_every_n_epochs, n_epochs_eval_model=args.n_epochs_eval_model))
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.num_iterations,
        accelerator="gpu" if args.cuda and torch.cuda.is_available() else "cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_check_interval if hasattr(args, 'val_check_interval') else 3,
    )

    # Train model
    trainer.fit(lit_model, train_loader)

if __name__ == '__main__':
    args = parse_args()
    main(args)