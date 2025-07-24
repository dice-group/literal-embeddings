import torch
import torch.nn as nn
import torch.nn.functional as F
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear


class LiteralEmbeddings(nn.Module):
    """
    A model for learning and predicting numerical literals using pre-trained KGE.

    Attributes:
        num_of_data_properties (int): Number of data properties (attributes).
        embedding_dims (int): Dimension of the embeddings.
        entity_embeddings (torch.tensor): Pre-trained entity embeddings.
        dropout (float): Dropout rate for regularization.
        gate_residual (bool): Whether to use gated residual connections.
        freeze_entity_embeddings (bool): Whether to freeze the entity embeddings during training.
    """

    def __init__(
        self,
        num_of_data_properties: int,
        embedding_dims: int,
        entity_embeddings: torch.tensor,
        dropout: float = 0.3,
        gate_residual=True,
        freeze_entity_embeddings=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dims
        self.num_of_data_properties = num_of_data_properties
        self.hidden_dim = embedding_dims * 2  # Combined entity + attribute embeddings
        self.gate_residual = gate_residual
        self.freeze_entity_embeddings = freeze_entity_embeddings


        # Use pre-trained entity embeddings
        self.entity_embeddings = nn.Embedding.from_pretrained(
            entity_embeddings.weight, freeze=self.freeze_entity_embeddings
        )

        #  data property (literal) embeddings
        self.data_property_embeddings = nn.Embedding(
            num_embeddings=num_of_data_properties,
            embedding_dim=self.embedding_dim,
        )

        # MLP components
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

        # Gated residual layer with layer norm
        self.gated_residual_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, entity_idx, attr_idx):
        """
        Args:
            entity_idx (Tensor): Entity indices (batch).
            attr_idx (Tensor): Attribute (Data property)  indices (batch).

        Returns:
            Tensor: scalar predictions.
        """
        # embeddings lookup
        e_emb = self.entity_embeddings(entity_idx)  # [batch, emb_dim]
        a_emb = self.data_property_embeddings(attr_idx)  # [batch, emb_dim]

        # Concatenate entity and property embeddings
        tuple_emb = torch.cat((e_emb, a_emb), dim=1)  # [batch, 2 * emb_dim]

        # MLP with dropout and ReLU
        z = self.dropout(
            F.relu(self.layer_norm(self.fc(tuple_emb)))
        )  # [batch, 2 * emb_dim]

        if self.gate_residual:
            # Gated residual logic (inline GLU)
            x_proj = self.gated_residual_proj(torch.cat((z, tuple_emb), dim=1))  # [batch, 4 * emb_dim]
            value, gate = x_proj.chunk(2, dim=-1)
            residual = value * torch.sigmoid(gate)
        else:
            residual = z + tuple_emb  # Simple residual

        # Output scalar prediction and flatten to 1D
        out = self.fc_out(residual).flatten()  # [batch]
        return out
    
    @property
    def device(self):
        return next(self.parameters()).device
    
class LiteralEmbeddingsClifford(nn.Module):
    """
    A model for learning and predicting numerical literals using pre-trained KGE with Clifford algebra layers.

    Attributes:
        num_of_data_properties (int): Number of data properties (attributes).
        embedding_dims (int): Dimension of the embeddings.
        entity_embeddings (torch.tensor): Pre-trained entity embeddings.
        dropout (float): Dropout rate for regularization.
        freeze_entity_embeddings (bool): Whether to freeze the entity embeddings during training.
    """

    def __init__(
        self,
        num_of_data_properties: int,
        embedding_dims: int,
        entity_embeddings: torch.tensor,
        dropout: float = 0.15,
        freeze_entity_embeddings=True,
        gate_residual=False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dims
        self.num_of_data_properties = num_of_data_properties
        self.hidden_dim = embedding_dims * 2  # Combined entity + attribute embeddings
        self.freeze_entity_embeddings = freeze_entity_embeddings
        self.gate_residual = gate_residual

        # Use pre-trained entity embeddings
        self.entity_embeddings = nn.Embedding.from_pretrained(
            entity_embeddings.weight, freeze=self.freeze_entity_embeddings
        )

        # data property (literal) embeddings
        self.data_property_embeddings = nn.Embedding(
            num_embeddings=num_of_data_properties,
            embedding_dim=self.embedding_dim,
        )
        # Clifford algebra parameters
        self.g = [-1, -1]
        self.n_blades = 2 ** len(self.g)  # 4 blades for 2D Clifford algebra
        self.in_channels = (
            self.embedding_dim * 2 // self.n_blades
        )  # Divide by number of blades

        # Clifford MLP components
        self.clif_1 = CliffordLinear(
            g=self.g,
            in_channels=self.in_channels,
            out_channels=self.hidden_dim,
            bias=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.out = CliffordLinear(
            g=self.g,
            in_channels=self.hidden_dim,
            out_channels=1,  # Output a single scalar prediction
            bias=True,
        )
        self.layer_norm = nn.LayerNorm([self.hidden_dim, self.n_blades])
        self.final_proj = nn.Linear(1 * self.n_blades, 1)

    def forward(self, entity_idx, attr_idx):
        """
        Args:
            entity_idx (Tensor): Entity indices (batch).
            attr_idx (Tensor): Attribute (Data property) indices (batch).

        Returns:
            Tensor: scalar predictions.
        """
        # embeddings lookup
        e_emb = self.entity_embeddings(entity_idx)  # [batch, emb_dim]
        a_emb = self.data_property_embeddings(attr_idx)  # [batch, emb_dim]

        # Concatenate entity and property embeddings
        tuple_emb = torch.cat((e_emb, a_emb), dim=1)  # [batch, 2 * emb_dim]
        
        # Reshape for Clifford algebra processing
        x = tuple_emb.view(-1, self.in_channels, self.n_blades)
        
        # Clifford linear transformation with layer norm and activation
        z = F.tanh(self.layer_norm(self.clif_1(x)))
        
        # Output layer - extract scalar component and flatten to 1D
        out = self.out(z)[:, :, 0].flatten()  # [batch]
        
        return out
    
    @property
    def device(self):
        return next(self.parameters()).device
