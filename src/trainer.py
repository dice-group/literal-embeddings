import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.utils import UncertaintyWeightedLoss
from src.model import LiteralEmbeddings
from src.utils import UncertaintyWeightedLoss


def train_literal_model(args, literal_dataset, kge_model, Literal_model=None):
    """
    Trains the Literal Embedding model standalone when a pre-trained KGE model is provided.

    Parameters:
    - args: Namespace with configuration values
    - literal_dataset: Dataset with literal triples
    - kge_model: Pretrained knowledge graph embedding model
    - Literal_model: The literal embedding model to train

    Returns:
    - Literal_model: Trained literal model
    - loss_log: Dictionary logging the literal training loss per epoch
    """

    device = args.device
    Literal_model = Literal_model.to(device)
    kge_model = kge_model.to(device)
    kge_model.eval()  # Freeze KGE model

    loss_log = {"lit_loss": []}
    optimizer = optim.Adam(Literal_model.parameters(), lr=args.lit_lr)

    # Prepare training data
    triples = literal_dataset.triples
    lit_entities = triples[:, 0].long().to(device)
    lit_properties = triples[:, 1].long().to(device)

    if args.multi_regression:
        num_samples = lit_entities.size(0)
        y_true = torch.zeros(
            num_samples, literal_dataset.num_data_properties, device=device
        )
        y_true[torch.arange(num_samples), lit_properties] = (
            literal_dataset.tails_norm.to(device)
        )
    else:
        y_true = literal_dataset.tails_norm.to(device)

    # Freeze gradients for KGE entity embeddings
    with torch.no_grad():
        ent_ebds = kge_model.entity_embeddings(lit_entities)

    # Can use a DataLoader for large datasets (currently assumes full-batch training)
    for epoch in (tqdm_bar := tqdm(range(args.lit_epochs))):
        Literal_model.train()
        optimizer.zero_grad()

        yhat = Literal_model(ent_ebds, lit_properties)
        lit_loss = F.l1_loss(yhat, y_true)

        lit_loss.backward()
        optimizer.step()

        tqdm_bar.set_postfix_str(f"loss_lit={lit_loss:.5f}")
        loss_log["lit_loss"].append(lit_loss.item())

    return Literal_model, loss_log


def train_model(
    model,
    train_dataloader,
    args,
    literal_dataset=None,
    Literal_model=None,
    val_dataloader=None,
):
    """
    Trains the model and logs the loss.

    Returns:
    - model: Trained KGE model
    - Literal_model: Trained literal model (if any)
    - loss_log: Dictionary of loss logs
    """
    device = args.device
    model.to(device)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    criterion = UncertaintyWeightedLoss()

    loss_log = {"ent_loss": []}
    if val_dataloader:
        loss_log["ent_loss_val"] = []

    if args.combined_training:
        # ====== Combined training (KGE + Literal) ======
        loss_log["lit_loss"] = []
        Literal_model.to(device)

        optimizer = optim.Adam(
            [
                {"params": model.parameters(), "lr": args.lr},
                {"params": Literal_model.parameters(), "lr": args.lit_lr},
                {"params": criterion.parameters(), "lr": args.lr},
            ]
        )

        for epoch in (tqdm_bar := tqdm(range(args.num_epochs))):
            model.train()
            Literal_model.train()
            ent_loss_total, lit_loss_total = 0.0, 0.0

            for batch in train_dataloader:
                train_X, train_y = batch
                train_X, train_y = train_X.to(device), train_y.to(device)

                # KGE model forward
                yhat_e = model(train_X)
                ent_loss = bce_loss_fn(yhat_e, train_y)

                # Literal model forward
                entity_ids = train_X[:, 0].long().to("cpu")
                lit_entities, lit_properties, y_true = literal_dataset.get_batch(
                    entity_ids, multi_regression=args.multi_regression
                )
                lit_entities, lit_properties, y_true = (
                    lit_entities.to(device),
                    lit_properties.to(device),
                    y_true.to(device),
                )

                ent_embeds = model.entity_embeddings(lit_entities)
                yhat_lit = Literal_model(
                    ent_embeds, lit_properties, train_ent_embeds=True
                )
                lit_loss = F.l1_loss(yhat_lit, y_true)

                # Combined loss
                # total_loss = criterion(ent_loss, lit_loss)
                total_loss = ent_loss + lit_loss
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                ent_loss_total += ent_loss.item()
                lit_loss_total += lit_loss.item()

            avg_ent_loss = ent_loss_total / len(train_dataloader)
            avg_lit_loss = lit_loss_total / len(train_dataloader)

            loss_log["ent_loss"].append(avg_ent_loss)
            loss_log["lit_loss"].append(avg_lit_loss)
            tqdm_bar.set_postfix_str(
                f"Avg. ent_loss={avg_ent_loss:.5f}, lit_loss={avg_lit_loss:.5f}"
            )

            # Optional: compute validation loss for KGE only
            if args.log_validation:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_X, val_y in val_dataloader:
                        val_X, val_y = val_X.to(device), val_y.to(device)
                        yhat_val = model(val_X)
                        val_loss += bce_loss_fn(yhat_val, val_y).item()
                avg_val_loss = val_loss / len(val_dataloader)
                loss_log["ent_loss_val"].append(avg_val_loss)

    else:
        # ====== KGE-only training ======
        optimizer = optim.Adam(
            [
                {"params": model.parameters(), "lr": args.lr},
                {"params": criterion.parameters(), "lr": args.lr},
            ]
        )

        for epoch in (tqdm_bar := tqdm(range(args.num_epochs))):
            model.train()
            total_loss = 0.0

            for batch in train_dataloader:
                train_X, train_y = batch
                train_X, train_y = train_X.to(device), train_y.to(device)

                yhat = model(train_X)
                loss = bce_loss_fn(yhat, train_y)
                loss = criterion(loss_ent=loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            loss_log["ent_loss"].append(avg_loss)
            tqdm_bar.set_postfix_str(f"loss_epoch={avg_loss:.5f}")

            # Optional validation
            if args.log_validation:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_X, val_y in val_dataloader:
                        val_X, val_y = val_X.to(device), val_y.to(device)
                        yhat_val = model(val_X)
                        val_loss += bce_loss_fn(yhat_val, val_y).item()
                avg_val_loss = val_loss / len(val_dataloader)
                loss_log["ent_loss_val"].append(avg_val_loss)

    return model, Literal_model, loss_log
