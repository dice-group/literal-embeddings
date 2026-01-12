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
            self.lambda_num = args.get("lambda_num", 0.1)
            

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.FloatTensor:
        h,r,t = x[:,0], x[:,1], t
        score = self.forward_k_vs_all(x)
        if self.use_literals and self.training:
            lit_score = self.score_literals(h, r)
            score += lit_score
        return score
        
        
        
    def score_literals(self, h_idx, r_idx):
        """
        Compute numeric literal scores for all tail entities given heads and relations.
        
        Args:
            h_idx: (B,) head entity indices
        
        Returns:
            Tensor of shape (B, num_entities): numeric scores for all tails
        """
        device = h_idx.device
        B = h_idx.size(0)
        num_entities = self.num_entities
        d = self.embedding_dim

        # ---- relation embeddings ----
        Wr = self.relation_embeddings(r_idx)  # (B, d)

        # ---- literals ----
        lit_ent = self.ea[:, 0].to(device)   # (L,)
        lit_attr = self.ea[:, 1].to(device)  # (L,)
        lit_val = self.v.to(device)          # (L,)

        attr_emb = self.attribute_embeddings(lit_attr)  # (L, d)
        u = lit_val.unsqueeze(1) * attr_emb             # (L, d)

        # ---- relation-conditioned scores per literal ----
        # scores[b, l] = u[l] dot Wr[b]
        scores = torch.einsum('bd,ld->bl', Wr, u)  # (B, L)

        # ---- masked softmax for head literals ----
        h_mask = (h_idx.unsqueeze(1) == lit_ent.unsqueeze(0))  # (B, L)
        masked_scores = scores.masked_fill(~h_mask, float('-inf'))
        att = torch.softmax(masked_scores, dim=1)  # (B, L)

        # ---- compute numeric summary per entity (vectorized) ----
        # expand lit_ent to (B, L) to match att
        lit_ent_batched = lit_ent.unsqueeze(0).expand(B, -1)  # (B, L)

        # Flatten batch for scatter_add
        att_flat = (att * lit_val.unsqueeze(0)).reshape(-1)        # (B*L,)
        lit_ent_flat = lit_ent_batched.reshape(-1)                 # (B*L,)
        batch_idx = torch.arange(B, device=device).repeat_interleave(lit_ent.size(0))  # (B*L,)

        # Create output tensor
        v_bar_all = torch.zeros(B, num_entities, device=device)
        v_bar_all.index_put_((batch_idx, lit_ent_flat), att_flat, accumulate=True)

        # ---- normalize attention per entity (vectorized) ----
        norm_flat = att.reshape(-1)  # (B*L,)
        norm_all = torch.zeros(B, num_entities, device=device)
        norm_all.index_put_((batch_idx, lit_ent_flat), norm_flat, accumulate=True)

        v_bar_all = v_bar_all / (norm_all + 1e-9)

        # ---- numeric scores for all tails ----
        v_bar_h = v_bar_all[torch.arange(B), h_idx].unsqueeze(1)  # (B, 1)
        E_num = torch.abs(v_bar_h - v_bar_all)                     # (B, num_entities)

        return -self.lambda_num * E_num  # (B, num_entities)



    

