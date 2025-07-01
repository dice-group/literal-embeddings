import torch
import torch.nn.functional as F
import torch.nn as nn
class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # output_dim * 2 because half for values, half for gates
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x_proj = self.proj(x)
        value, gate = x_proj.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)

class LiteralEmbeddings(torch.nn.Module):
    def __init__(
        self,
        num_of_data_properties: int = None,
        dropout: float = 0.3,
        embedding_dims: int = None,
        multi_regression=False,
    ):
        super().__init__()
        self.embeddings_dim = embedding_dims
        self.num_of_data_properties = num_of_data_properties
        self.multi_regressor = multi_regression
        self.out_features = self.num_of_data_properties if self.multi_regressor else 1

        self.data_property_embeddings = torch.nn.Embedding(
            num_embeddings=num_of_data_properties, embedding_dim=self.embeddings_dim
        )

        self.hidden_dim = self.embeddings_dim *2
        self.fc1 = nn.Linear(self.hidden_dim , self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.gate_fc  = GatedLinearUnit(self.hidden_dim*2)
        self.norm1 = torch.nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, relation_idx, train_ent_embeds=False):

        head_entity_embeddings = x
        if not train_ent_embeds:
            head_entity_embeddings = x.detach()

        a_emb = self.data_property_embeddings(relation_idx)
        tuple_embeddings = torch.cat(( head_entity_embeddings, a_emb ), dim=1)
        out1 = self.dropout(self.norm1(F.relu(self.fc1(tuple_embeddings))))
        gated_residual =self.gate_fc(torch.cat((out1, tuple_embeddings), dim = 1))
        out = self.fc_out(gated_residual)
        if not self.multi_regressor:
            out = out.flatten()
        return out
