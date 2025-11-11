import os
import torch

from tqdm import tqdm
from dicee.evaluator import Evaluator
from dicee.knowledge_graph import KG
from torch.utils.data import DataLoader
from dicee.static_funcs import  read_or_load_kg, store
from src.static_funcs import  save_kge_experiments, get_full_storage_path
from src.static_train_utils import   get_ff_models
from src.dataset import NegSampleDataset

def train_kge_ff(args):
    """Train a KGE model with ff-update."""
    # Set up experiment storage path
    args.learning_rate = args.lr

    assert args.scoring_technique == "NegSample" , "Forward_forward ony supports neg-sampling for now"
    args.full_storage_path = get_full_storage_path(args)
    os.makedirs(args.full_storage_path, exist_ok=True)
    print("Training dir", args.full_storage_path)


    # Device and seed setup
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # Load dataset and set entity/relation counts
    entity_dataset = read_or_load_kg(args, KG)
    args.num_entities = entity_dataset.num_entities
    args.num_relations = entity_dataset.num_relations

    # train dataloder, valid_dataloader
    train_dataset = NegSampleDataset(
        neg_sample_ratio=2,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        train_set=entity_dataset.train_set
    )
    train_dataloader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_core
)
    supported_models = ["DistMult", "CLNN", "FFKGE"]
    assert args.model in supported_models , f"{args.model} not supported for forward-forward training"

    kge_model = get_ff_models(args)
    kge_model.to(args.device)

    optimizer = torch.optim.Adam(kge_model.parameters(), lr=args.lr)
    for epoch in (tqdm_bar := tqdm(range(args.num_epochs))):
        kge_model.train()
        epoch_loss = 0
        for batch in train_dataloader:
            train_X, train_y = batch
            train_X = train_X.to(args.device)
            x_pos = train_X[:, 0]
            x_neg = train_X[:, 1]
            batch_report = kge_model.ff_update(x_pos, x_neg, optimizer)
            epoch_loss += batch_report["loss"]
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        tqdm_bar.set_postfix_str(f" ff_loss={avg_epoch_loss:.5f}")


    evaluator = Evaluator(args=args)
    kge_model.to("cpu")
    evaluator.eval(
        dataset=entity_dataset,
        trained_model=kge_model,
        form_of_labelling="EntityPrediction",
    )
    print(f"Forward-forward  training for the Model {args.model} completed.")
    save_kge_experiments(args=args, loss_log=None)

