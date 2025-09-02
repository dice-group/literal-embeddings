import torch
import numpy as np
import pandas as pd

import torch.nn.functional as F
from dicee.models import Keci
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear

def load_num_lit(ent2idx, dataset_dir, lit_norm='min_max', wrap=True):
    literal_df = pd.read_csv(f'{dataset_dir}/literals/numerical_literals.txt', names=['ent', "attribute", "value"], sep='\t')
    attr2idx = {v: k for k, v in enumerate(literal_df['attribute'].unique())}
    literal_df["ent_idx"] = literal_df["ent"].map(ent2idx)
    literal_df["attr_idx"] = literal_df["attribute"].map(attr2idx)
    literal_df["value_norm"] = literal_df.groupby("attribute")["value"].transform(
                lambda x: (x - x.mean()) / x.std()
    )
    
    return literal_df, attr2idx



class Lit_Keci(Keci):
    def __init__(self, args, ent2idx, rel2idx):
        super().__init__(args)
        self.use_literals = args["use_literals"]
        if self.use_literals:
            #### begin literals
            self.entity_to_idx = ( ent2idx if isinstance(ent2idx, dict)
                else {v: i for i, v in ent2idx["entity"].items()} )
            self.literal_df, self.attr2idx = load_num_lit(self.entity_to_idx, args["dataset_dir"])
            self.ea =  torch.tensor(self.literal_df[["ent_idx", "attr_idx"]].values, dtype=torch.long)
            self.v = torch.tensor(self.literal_df["value_norm"].values, dtype=torch.float32)

            self.attribute_embeddings = torch.nn.Embedding( num_embeddings=len(self.attr2idx), 
                                                               embedding_dim=self.embedding_dim)
            # Clifford algebra parameters
            self.g = [-1]
            self.n_blades = 2 ** len(self.g)  # 4 blades for 2D Clifford algebra
            
            self.in_channels = (self.embedding_dim * 2 // self.n_blades)  # Divide by number of blades

            self.clif_lit_in = CliffordLinear(g=self.g,in_channels=self.in_channels, out_channels=self.in_channels, bias=True)
            self.clif_lit_out = CliffordLinear(g=self.g,in_channels=self.in_channels, out_channels=1, bias=True)
            self.clif_layer_norm = torch.nn.LayerNorm([self.in_channels, self.n_blades])

    def forward_k_vs_all(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.FloatTensor:
        if self.use_literals and self.training:
            # (1) Gather the set of entity IDs used in this batch
            batch_ents = torch.unique(torch.cat((x[:, 0], t), dim=0))
            self.ea, self.v = self.ea.to(batch_ents.device), self.v.to(batch_ents.device)

            # (2) Pull the (entity, attribute) rows that belong to the batch
            mask = torch.isin(self.ea[:, 0], batch_ents)
            batch_ea, batch_vals = self.ea[mask], self.v[mask]          # shape: (B, 2) , (B,)

            # (3) Embed the entities and fetch attribute indices
            ent_emb   = self.entity_embeddings(batch_ea[:, 0])        # (B, D)
            attr_idx  = batch_ea[:, 1].long()                         # (B,)

            # (4) Compute the literal loss in a single line
            lit_loss = F.l1_loss(self.score_literals(ent_emb, attr_idx), batch_vals)
            
            head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
            E = self.entity_embeddings.weight
            k_vs_all_score =  super().k_vs_all_score(head_ent_emb, rel_ent_emb, E)
            return k_vs_all_score, lit_loss
        else:
            head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
            E = self.entity_embeddings.weight
            k_vs_all_score =  super().k_vs_all_score(head_ent_emb, rel_ent_emb, E)
            return k_vs_all_score
        
        
    def score_literals(self, e_emb, attr_idx):
        a_emb = self.attribute_embeddings(attr_idx)
        tuple_emb = torch.cat((e_emb, a_emb), dim=1)  # [batch, 2 * emb_dim]
        # Reshape for Clifford algebra processing
        x = tuple_emb.view(-1, self.in_channels, self.n_blades)
        x_hid = self.clif_lit_in(x)
        lit_score = self.clif_lit_out(x_hid)[:,:,0].flatten()
        return lit_score