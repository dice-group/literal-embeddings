import torch
import torch.nn.functional as F


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

        self.fc1 = torch.nn.Linear(
            in_features=self.embeddings_dim * 2,
            out_features=self.embeddings_dim * 2,
            bias=True,
        )
        self.fc2 = torch.nn.Linear(
            in_features=self.embeddings_dim * 2,
            out_features=self.out_features,
            bias=True,
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, relation_idx, train_ent_embeds=False):

        head_entity_embeddings = x
        if not train_ent_embeds:
            head_entity_embeddings = x.detach()

        relation_embeddings = self.data_property_embeddings(relation_idx)
        tuple_embeddings = torch.cat(
            (head_entity_embeddings, relation_embeddings), dim=1
        )

        out1 = F.relu(self.fc1(tuple_embeddings))
        out1 = self.dropout(out1)
        out2 = self.fc2(out1 + tuple_embeddings)
        if not self.multi_regressor:
            out2 = out2.flatten()
        return out2
