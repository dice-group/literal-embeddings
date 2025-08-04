import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


def train_literal_model(args, kge_model,
                        literal_model=None, literal_batch_loader=None):
    """
    Trains the Literal Embedding model standalone when a pre-trained KGE model is provided.

    Parameters:
    - args: Namespace with configuration values
    - kge_model: Pretrained knowledge graph embedding model
    - literal_model: The literal embedding model to train
    - literal_batch_loader: Dataloader for literal triples

    Returns:
    - literal_model: Trained literal model
    - loss_log: Dictionary logging the literal training loss per epoch
    """

    device = getattr(args, "device", torch.device("cpu"))
    literal_model = literal_model.to(device)
    kge_model = kge_model.to(device)
    kge_model.eval()  # Freeze KGE model

    loss_log = {"lit_loss": []}
    optimizer = optim.Adam(literal_model.parameters(), lr=args.lit_lr)

    # Can use a DataLoader for large datasets (currently assumes full-batch training)
    # Training loop
    for epoch in (tqdm_bar := tqdm(range(args.lit_epochs))):
        epoch_loss = 0
        for batch_x, batch_y in literal_batch_loader:
            optimizer.zero_grad()
            lit_entities = batch_x[:, 0].long().to(device)
            lit_properties = batch_x[:, 1].long().to(device)
            batch_y = batch_y.to(device)
            yhat = literal_model(lit_entities, lit_properties)
            lit_loss = F.l1_loss(yhat, batch_y)
            lit_loss.backward()
            optimizer.step()
            epoch_loss += lit_loss.item()

        avg_epoch_loss = epoch_loss / len(literal_batch_loader)
        tqdm_bar.set_postfix_str(f"loss_lit={avg_epoch_loss:.5f}")
        loss_log["lit_loss"].append(avg_epoch_loss)

    return literal_model, loss_log