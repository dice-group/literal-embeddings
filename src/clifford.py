import torch
import torch.nn as nn
import torch_scatter

import numpy as np
import pandas as pd

import torch.nn.functional as F
from dicee.models.base_model import BaseKGE
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear




import torch
import torch.nn as nn
import torch.nn.functional as F

def batched_kronecker(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kronecker product of two batched vectors.

    Args:
        x: Tensor of shape (B, N)
        y: Tensor of shape (B, N)

    Returns:
        Tensor of shape (B, N*N)
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("Inputs must be 2D tensors of shape (B, N)")
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape")

    B, N = x.shape

    # (B, N, 1) * (B, 1, N) → (B, N, N)
    kron = x.unsqueeze(2) * y.unsqueeze(1)

    # Flatten last two dims
    return kron.reshape(B, N * N)


class CLNN_KGE(BaseKGE):
    def __init__(self, args, edge_index=None, edge_type=None):
        super().__init__(args)
        self.name = "CLNN_KGE"

        self.edge_index = edge_index
        self.edge_type = edge_type

        # Clifford algebra parameters
        self.g = [-1]
        self.n_blades = 2 ** len(self.g)  # for g=[-1], n_blades=2

        if self.embedding_dim % self.n_blades != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by n_blades ({self.n_blades})."
            )

        # base channels per blade
        self.C = self.embedding_dim // self.n_blades

        # MLP input channels after concatenating h and r in hypercomplex form:
        # emb_h: (B, C, n_blades), emb_r: (B, C, n_blades) -> cat along channel dim -> (B, 2C, n_blades)
        self.in_channels = self.C * self.C *2

        

        # Clifford MLP that produces a query q_hr in the same space as entity embeddings (C, n_blades)
        self.clif_in = CliffordLinear(g=self.g, in_channels=self.in_channels, out_channels=self.C , bias=True)
        self.clif_out = CliffordLinear(g=self.g, in_channels=self.C , out_channels=self.C, bias=True)

        self.clif_layer_norm = nn.LayerNorm([self.C , self.n_blades])

    def to_hypercomplex(self, x: torch.Tensor, n_components: int) -> torch.Tensor:
        """
        x: (..., n_components * d)
        returns: (..., d, n_components)
        """
        if x.shape[-1] % n_components != 0:
            raise ValueError(f"Last dimension ({x.shape[-1]}) not divisible by {n_components}")
        return x.reshape(*x.shape[:-1], n_components, -1).transpose(-2, -1)


    def k_vs_all_score(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.Tensor:
        """
        emb_h: (B, embedding_dim)  (flat)
        emb_r: (B, embedding_dim)  (flat)
        Returns:
            scores: (B, num_entities)
        """
        # 1) reshape to hypercomplex
        # h_hc = self.to_hypercomplex(emb_h, n_components=self.n_blades)  # (B, C, n_blades)
        # r_hc = self.to_hypercomplex(emb_r, n_components=self.n_blades)  # (B, C, n_blades)

        x = batched_kronecker(emb_h, emb_r)
        x = self.to_hypercomplex(x, n_components=self.n_blades)

        #x = torch.cat([h_hc, r_hc, hr], dim=1)

        # 2) Clifford-MLP to produce query q_hr
        #x = torch.cat([h_hc, r_hc], dim=1)          # (B, 2C, n_blades)
        q = self.clif_in(x)                         # (B, C, n_blades)
        q = self.clif_layer_norm(q)
        q = F.relu(q)
        q = self.clif_out(q)                        # (B, C, n_blades)

        # 3) get all entity embeddings as hypercomplex      # (E, embedding_dim)
        ent_hc = self.to_hypercomplex(self.entity_embeddings.weight, n_components=self.n_blades)  # (E, C, n_blades)

        # 4) KvsAll score via inner product in hypercomplex space
        # scores[b, e] = sum_{c,blade} q[b,c,blade] * ent[e,c,blade]
        scores = torch.einsum("bci,eci->be", q, ent_hc)  # (B, E)

        return scores

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
        emb_head, emb_rel = self.get_head_relation_representation(x)  # expected flat (B, embedding_dim)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        # (b,d),     (b,d)
        raise NotImplementedError("forward_k_vs_sample method is not implemented in TapireCL model.")


    def score(self, h, r, t):
        raise NotImplementedError("score method is not implemented in CLNN model.")
    


def to_hypercomplex( x, n_components):
    """
    x: (..., n_components * d)
    returns: (..., d, n_components)
    """
    if x.shape[-1] % n_components != 0:
        raise ValueError(
            f"Last dimension ({x.shape[-1]}) not divisible by {n_components}"
        )

    return (
        x.reshape(*x.shape[:-1], n_components, -1)
        .transpose(-2, -1)
    )

def scatter_sum(messages, dst, num_nodes):
    """
    messages: (E, C, n_blades)
    dst: (E,) long tensor with 0 <= dst < num_nodes
    num_nodes: int

    returns: (num_nodes, C, n_blades)
    """
    out = messages.new_zeros((num_nodes, messages.size(1), messages.size(2)))
    out.index_add_(0, dst, messages)  # sums messages into rows indexed by dst
    return out

import torch




class CliffordMessagePassing(nn.Module):
    def __init__(self, g, emb_dim):
        super().__init__()

        self.g = g
        self.n_blades = 2 ** len(g)
        self.C = emb_dim // self.n_blades

        self.msg_linear = CliffordLinear(
            g=g,
            in_channels=self.C,
            out_channels=self.C,
            bias=True
        )

        self.update_linear = CliffordLinear(
            g=g,
            in_channels=self.C,
            out_channels=self.C,
            bias=True
        )
        # Relation gating to modulate messages per edge.
        self.gate_linear = nn.Linear(self.C * self.n_blades, 1, bias=True)

    def forward(self, h, edge_index, edge_type, relation_emb):
        """
        h: (num_nodes, emb_dim)          # standard embedding
        relation_emb: (num_relations, emb_dim)
        edge_index: (2, num_edges)
        edge_type: (num_edges,)
        to_hypercomplex: callable that reshapes flat embeddings
        """

        # 1. reshape embeddings into Clifford hypercomplex
        h_hc = to_hypercomplex(h, self.n_blades)                 # (num_nodes, C, n_blades)
        r_hc = to_hypercomplex(relation_emb, self.n_blades)      # (num_relations, C, n_blades)

        # 2. index edges
        edge_index = edge_index.long().to(h.device)
        src, dst = edge_index
        h_src = h_hc[src]           # (num_edges, C, n_blades)
        r_uv  = r_hc[edge_type]     # (num_edges, C, n_blades)

        # 3. compute messages
        m = self.msg_linear(h_src * r_uv)
        gate = torch.sigmoid(self.gate_linear(r_uv.reshape(r_uv.size(0), -1)))
        m = m * gate.view(-1, 1, 1)
        # ensure correct dtype and device
        dst = dst.long().to(h_hc.device)


        # 4. aggregate messages with degree normalization
        num_nodes = h_hc.size(0)
        deg_src = torch.bincount(src, minlength=num_nodes).to(m.device)
        deg_dst = torch.bincount(dst, minlength=num_nodes).to(m.device)
        norm = (deg_src[src] * deg_dst[dst]).clamp(min=1).pow(-0.5)
        m = m * norm.view(-1, 1, 1)
        h_agg = torch_scatter.scatter_sum(
            src=m,
            index=dst,
            dim=0,
            dim_size=num_nodes
        )

        # 5. update node embeddings
        h_out = self.update_linear(h_hc + h_agg)

        # 6. flatten back to (num_nodes, emb_dim) if needed
        h_out = h_out.contiguous()
        h_out_flat = h_out.view(h_out.size(0), -1)
        return h_out_flat

   
        
# class CLNN_KGE(BaseKGE):
#     def __init__(self, args, edge_index=None, edge_type=None):
#         super().__init__(args)
#         self.name = "CLNN_KGE"
#         self.edge_index = edge_index
#         self.edge_type = edge_type
#         # Clifford algebra parameters
#         self.g = [-1]
#         self.n_blades = 2 ** len(self.g)  # 4 blades for 2D Clifford algebra

#         # message passing layers
#         self.num_gnn_layers = 2
#         self.gnn_layers = nn.ModuleList(
#             [CliffordMessagePassing(g=self.g, emb_dim=self.embedding_dim) for _ in range(self.num_gnn_layers)]
#         )
#         self.gnn_norms = nn.ModuleList(
#             [nn.LayerNorm(self.embedding_dim) for _ in range(self.num_gnn_layers)]
#         )
#         self.in_channels = (self.embedding_dim * 2 // self.n_blades)  # Divide by number of blades
#         # self.num_clifford_layers = 2
#         self.out = CliffordLinear(g=self.g,in_channels=self.in_channels, out_channels=self.num_entities, bias=True)
#         self.clif_layer_norm = torch.nn.LayerNorm([self.in_channels, self.n_blades])




#     def score_k_vs_all(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.FloatTensor:
#         emb_h = to_hypercomplex(emb_h, n_components=self.n_blades)
#         emb_r = to_hypercomplex(emb_r, n_components=self.n_blades)

#         enb_cat = torch.cat([emb_h, emb_r], dim=1)  # (B, 2C, n_blades)
#         x = self.clif_layer_norm(enb_cat)
#         scores = self.out(x)[:,:,0]  # (B, E), return only scalar part
#         # # 1. Clifford product: relation-conditioned query
#         # query = emb_h * emb_r                # (B, C, n_blades)

#         # # 2. Expand for broadcasting
#         # query = query.unsqueeze(1)   # (B, 1, C, n_blades)
#         # all_entities = to_hypercomplex(self.entity_embeddings.weight.data, self.n_blades).unsqueeze(0)  # (1, E, C, n_blades)

#         # # 3. Compute squared distance
#         # scores = -((query - all_entities) ** 2).sum(dim=(2, 3))  # (B, E)

#         return scores
        

#     def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
#         emb_head = self.entity_embeddings.weight.data
#         emb_rel = self.relation_embeddings.weight.data
#         h_ent = emb_head
#         for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
#             h_ent = norm(h_ent + gnn(h_ent, self.edge_index, self.edge_type, emb_rel))
#         emb_head_updated = h_ent[x[:,0]]
#         emb_rel_batch = emb_rel[x[:,1]]
#         return self.score_k_vs_all(emb_h=emb_head_updated, emb_r=emb_rel_batch)

#     def score(self, h, r, t):
#         raise NotImplementedError("score method is not implemented in CLNN_KGE model.")
#         # Clifford inner product
#         return -(h * r - t).pow(2).sum(dim=(1,2))

    
        
