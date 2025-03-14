import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import HuberLoss
from tqdm import tqdm

from src.model import LiteralEmbeddings


def train_literal_model(
    args,
    literal_dataset: None,
    kge_model: None,
):
    """
    Method to train Literal Embedding model standalone when a Pre-trained model is provided
    """
    device = args.device
    kge_model = kge_model.to(args.device)
    loss_log = {"lit_loss": []}

    Literal_model = LiteralEmbeddings(
        num_of_data_properties=literal_dataset.num_data_properties,
        embedding_dims=args.embedding_dim,
        multi_regression=args.multi_regression,
    ).to(args.device)

    lit_entities = literal_dataset.train_data["triples"][:, 0].long()
    lit_properties = literal_dataset.train_data["triples"][:, 1].long()

    if args.multi_regression:
        # Create a zero tensor of shape [num_samples, num_relations]
        num_samples = lit_entities.shape[0]  # Number of rows in dataset
        y_true = torch.full(
            (num_samples, literal_dataset.num_data_properties), 0, dtype=torch.float32
        )  # Match dtype with `v`

        # Assign tail values at the correct relation indices per row
        y_true[torch.arange(num_samples), lit_properties] = literal_dataset.train_data[
            "tails_norm"
        ]  # Uses row index from torch

    else:
        y_true = literal_dataset.train_data["tails_norm"]

    ent_ebds = kge_model.entity_embeddings(lit_entities)

    optimizer = optim.Adam(Literal_model.parameters(), lr=args.lit_lr)
    for epoch in (tqdm_bar := tqdm(range(args.lit_epochs))):
        yhat = Literal_model(ent_ebds, lit_properties)
        lit_loss = F.l1_loss(yhat, y_true)
        lit_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tqdm_bar.set_postfix_str(f" loss_lit={lit_loss:.5f} ")
        loss_log["lit_loss"].append(lit_loss.item())

    return Literal_model, loss_log


def train_model(
    model,
    train_dataloader,
    args,
    literal_dataset=None,
    Literal_model=None,
):
    """
    Trains the model and logs the loss.

    Parameters:
    - model: (Any ) Knowledge Graph Embedding model instance
    - train_dataloader: Dataloader for training data
    - args: Configuration arguments
    - literal_dataset: Literal dataset (optional)
    - Literal_model: Literal model instance (optional)
    - optimize_with_literal: Flag to optimize with literals

    Returns:
    - loss_log: Dictionary of loss logs
    """
    device = args.device

    loss_log = {"ent_loss": []}
    y_true, lit_entities, lit_properties = None, None, None
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    if args.combined_training:

        # intialize loss logs and optimizers for Literal and KGE odel
        loss_log["lit_loss"] = []
        optimizer = optim.Adam(
            [
                {"params": model.parameters(), "lr": args.lr},
                {"params": Literal_model.parameters(), "lr": args.lit_lr},
            ]
        )
    else:
        # optimzer when only KGE model is being trained
        optimizer = optim.Adam(
            [
                {"params": model.parameters(), "lr": args.lr},
            ]
        )

    for epoch in (tqdm_bar := tqdm(range(args.num_epochs))):
        ent_loss = 0
        lit_loss = 0
        model.train()

        if args.combined_training:
            Literal_model.to(device)
            Literal_model.train()
            for batch in train_dataloader:

                # begin entity emebedding model training
                train_X, train_y = batch
                train_X, train_y = train_X.to(device), train_y.to(device)
                # ent_loss_batch = model.training_step(batch=(train_X, train_y))
                yhat_e = model(train_X)
                ent_loss_batch = bce_loss_fn(yhat_e, train_y)

                # prepare batch for  literal embedding model training
                # Build literal triples based on entity triples
                # Extract data properties from entity triple (train_X)

                batch_literal_entity_indices = train_X[:, 0].long().to("cpu")

                # batch_literal_entity_indices = batch_literal_entity_indices.to(device)

                lit_entities, lit_properties, y_true = literal_dataset.get_batch(
                    batch_literal_entity_indices, multi_regression=args.multi_regression
                )
                lit_entities, lit_properties, y_true = (
                    lit_entities.to(device),
                    lit_properties.to(device),
                    y_true.to(device),
                )
                ent_ebds = model.entity_embeddings(lit_entities)
                yhat = Literal_model.forward(ent_ebds, lit_properties)
                lit_loss_batch = F.l1_loss(yhat, y_true)

                # begin combined loss procedure

                # Define weights for each loss
                w1, w2 = args.alpha, args.beta
                batch_loss = (w1 * ent_loss_batch) + (w2 * lit_loss_batch)
                # batch_loss = combine_losses(ent_loss_batch, lit_loss_batch)
                # backward loss and optimization step
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ent_loss += ent_loss_batch
                lit_loss += lit_loss_batch

            avg_epoch_loss_ent = ent_loss / len(train_dataloader)
            avg_epoch_loss_lit = lit_loss / len(train_dataloader)
            tqdm_bar.set_postfix_str(
                f" Avg. loss_ent={avg_epoch_loss_ent:.5f} , loss_lit={avg_epoch_loss_lit:.5f} "
            )

            loss_log["ent_loss"].append(avg_epoch_loss_ent.item())
            loss_log["lit_loss"].append(avg_epoch_loss_lit.item())

        else:
            # trainning step only for KGE model ( without Literal Model)
            for batch in train_dataloader:
                train_X, train_y = batch
                train_X, train_y = train_X.to(device), train_y.to(device)
                yhat_e = model(train_X)
                ent_loss_batch = bce_loss_fn(yhat_e, train_y)
                ent_loss += ent_loss_batch
                ent_loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            tqdm_bar.set_postfix_str(f" loss_epoch={ent_loss:.5f}")
            avg_epoch_loss = ent_loss / len(train_dataloader)
            loss_log["ent_loss"].append(avg_epoch_loss.item())

    return loss_log
