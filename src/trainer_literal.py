import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

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
    #kge_model = kge_model.to(device)
    kge_model.eval()  # Freeze KGE model

    loss_log = {"lit_loss": []}
    optimizer = optim.Adam(Literal_model.parameters(), lr=args.lit_lr)

    # Prepare training data
    triples = literal_dataset.triples
    lit_entities = triples[:, 0].long()
    lit_properties = triples[:, 1].long()

    if args.multi_regression:
        num_samples = lit_entities.size(0)
        y_true = torch.zeros(
            num_samples, literal_dataset.num_data_properties
        )
        y_true[torch.arange(num_samples), lit_properties] = (
            literal_dataset.tails_norm
        )
    else:
        y_true = literal_dataset.tails_norm

    # Freeze gradients for KGE entity embeddings
    with torch.no_grad():
        ent_ebds = kge_model.entity_embeddings(lit_entities)

    ent_ebds, lit_properties, y_true = ent_ebds.to(device), lit_properties.to(device), y_true.to(device)

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