import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from config import parse_args
from callbacks import ASWA

# Set float32 matmul precision for better performance
torch.set_float32_matmul_precision('medium')

from dataset import KGDataset
from model import *
from trainer import LitModel
from evaluate import Evaluator
import json
from pytorch_lightning.callbacks.stochastic_weight_avg import \
    StochasticWeightAveraging as SWA

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
    exp_dir = f"Experiments/{args.dataset}_{args.model}/"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        
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
        
    # Initialize model
    lit_model = LitModel(
        args.model, train_dataset, args.edim, args.rdim, kwargs,
        args.lr, args.label_smoothing, model_mapping=model_mapping,
        er_vocab=er_vocab, evaluator=evaluator, exp_dir = exp_dir   
    )
    
    # Setup callbacks
    callbacks  = []
    args.adaptive_swa = True
    if args.swa:
        callbacks.append(SWA(swa_epoch_start=1, swa_lrs=args.lr))
        print("Using Stochastic Weight Averaging (SWA)")
    elif args.adaptive_swa:
        callbacks.append(ASWA(path=exp_dir, num_epochs=args.num_iterations))
        print("Using Adaptive Stochastic Weight Averaging (ASWA)") 
    else:
        print("Not using weight averaging")
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.num_iterations,
        accelerator="gpu" if args.cuda and torch.cuda.is_available() else "cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar= True,callbacks=callbacks,
    )

    # Train model
    trainer.fit(lit_model, train_loader, test_loader)
    
    # Save configuration
    config_to_save = vars(args)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_to_save, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)