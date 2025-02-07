import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import HuberLoss
from tqdm import tqdm

from src.model import LiteralEmbeddings
from src.utils import combine_losses


def train_literal_model(args, literal_dataset: None, kge_model: None, device: None):
    """
    Method to train Literal Embedding model standalone when a Pre-trained model is provided"""

    kge_model = kge_model.to(device)

    Literal_model = LiteralEmbeddings(
        num_of_data_properties=literal_dataset.num_data_properties,
        embedding_dims=args.embedding_dim,
    ).to(device)

    lit_y = torch.FloatTensor(literal_dataset.train_df["normalized_tail"].tolist()).to(
        device
    )
    lit_entities = torch.LongTensor(literal_dataset.train_df["head_idx"].values).to(
        device
    )
    lit_properties = torch.LongTensor(literal_dataset.train_df["rel_idx"].values).to(
        device
    )

    ent_ebds = kge_model.entity_embeddings(lit_entities)

    optimizer = optim.Adam(Literal_model.parameters(), lr=args.lit_lr)

    for epoch in (tqdm_bar := tqdm(range(200))):
        yhat = Literal_model(ent_ebds, lit_properties)
        lit_loss = F.l1_loss(yhat, lit_y)
        lit_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tqdm_bar.set_postfix_str(f" loss_lit={lit_loss:.5f} ")

    return Literal_model


def train_model(
    model,
    train_dataloader,
    args,
    device,
    literal_dataset=None,
    Literal_model=None,
    random_literals=True,
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

    loss_log = {"ent_loss": []}
    lit_y, lit_entities, lit_properties = None, None, None
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    if args.optimize_with_literals:

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
        Literal_model.train()

        if args.optimize_with_literals:
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

                batch_literals = literal_dataset.lit_value_tensor[
                    batch_literal_entity_indices
                ].to(device)

                batch_size, num_data_props_batch = batch_literals.shape
                batch_literal_entity_indices = batch_literal_entity_indices.to(device)

                #randomly sample data properties
                if random_literals:
                    random_data_props = torch.randint(
                        0, num_data_props_batch, (batch_size,)
                    ).to(device)
                    y_vals = (
                        batch_literals.gather(1, random_data_props.view(-1, 1))
                        .squeeze(1)
                    )
                    ent_ebds = model.entity_embeddings(batch_literal_entity_indices)
                    yhat = Literal_model.forward(ent_ebds, random_data_props)
                    lit_loss_batch = F.l1_loss(yhat, y_vals)
                    
                else:
                    # combine the epoch loss on all the data proeprties 
                    # if an entity does not have some literal values, it is set to 0
                    ent_ebds = model.entity_embeddings(batch_literal_entity_indices)

                    #average literal loss on all data properties
                    lit_loss_batch = (
                        sum(
                            F.l1_loss(
                                Literal_model.forward(
                                    ent_ebds, torch.full((batch_size,), i)
                                ),
                                batch_literals[:, i].to(device),
                            )
                            for i in range(num_data_props_batch)
                        )
                        / num_data_props_batch
                    )

                # begin combined loss procedure

                # Define weights for each loss
                # w1, w2 = 0.7, 0.3
                # batch_loss = (w1 * ent_loss_batch) + (w2 * lit_loss_batch)
                batch_loss = combine_losses(ent_loss_batch, lit_loss_batch)
                

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

                ent_loss_batch = model.training_step(batch=(train_X, train_y))
                ent_loss += ent_loss_batch
                ent_loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            tqdm_bar.set_postfix_str(f" loss_epoch={ent_loss:.5f}")
            avg_epoch_loss = ent_loss / len(train_dataloader)
            loss_log["ent_loss"].append(avg_epoch_loss.item())

    return loss_log
