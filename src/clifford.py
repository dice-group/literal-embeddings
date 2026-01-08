import torch
import numpy as np
import pandas as pd

import torch.nn.functional as F
from dicee.models.base_model import BaseKGE
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear




class CLNN_KGE(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = "CLNN_KGE"
        # Clifford algebra parameters
        self.g = [-1]
        self.n_blades = 2 ** len(self.g)  # 4 blades for 2D Clifford algebra
        
        self.in_channels = (self.embedding_dim * 2 // self.n_blades)  # Divide by number of blades

        self.clif_in = CliffordLinear(g=self.g,in_channels=self.in_channels, out_channels=self.in_channels, bias=True)
        self.clif_lit_out = CliffordLinear(g=self.g,in_channels=self.in_channels, out_channels=self.num_entities, bias=True)
        self.clif_layer_norm = torch.nn.LayerNorm([self.in_channels, self.n_blades])

    def k_vs_all_score(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.FloatTensor:
        cat_tensor = torch.cat([emb_h, emb_r], dim=-1)
        # Reshape for Clifford algebra processing
        x = cat_tensor.view(-1, self.in_channels, self.n_blades)
        x =  self.clif_in(x)
        x = self.clif_layer_norm(x)
        x = F.relu(x)
        x = self.clif_lit_out(x)
        return x[:,:,0]  # Return only the scalar part

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        emb_head, emb_rel = self.get_head_relation_representation(x)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        # (b,d),     (b,d)
        raise NotImplementedError("forward_k_vs_sample method is not implemented in TapireCL model.")


    def score(self, h, r, t):
        raise NotImplementedError("score method is not implemented in TapireCL model.")
        
        
        
   