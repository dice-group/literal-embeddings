import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear
from dicee.models.base_model import BaseKGE
from src.clifford_utils import IdentityBladeAlgebra, MultiVectorAct
from src.clifford_utils import  clifford_multiply, hypercomplex_multiply, to_hypercomplex
try:
    from cliffordlayers.nn.modules.batchnorm import CliffordBatchNorm1d
except Exception:  # pragma: no cover
    CliffordBatchNorm1d = None

class CLNN_KGE(BaseKGE):
    def __init__(self, args, edge_index=None, edge_type=None, entity2idx=None):
        super().__init__(args)
        self.name = "CLNN_KGE"

        self.edge_index = edge_index
        self.edge_type = edge_type

        # Use a complex-like 2-blade setup by default.
        self.g = [-1]
        self.n_blades = 2 ** len(self.g)
        if self.embedding_dim % self.n_blades != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by n_blades ({self.n_blades})."
            )

        self.C = self.embedding_dim // self.n_blades

        # KvsAll scorer (query MLP over [h, r]).
        self.clif_in = CliffordLinear(g=self.g, in_channels=2 * self.C, out_channels=self.C, bias=True)
        self.clif_out = CliffordLinear(g=self.g, in_channels=self.C, out_channels=self.C, bias=True)
        self.skip_proj = CliffordLinear(g=self.g, in_channels=2 * self.C, out_channels=self.C, bias=True)
        self.res_gate = nn.Parameter(torch.tensor(0.0))
        if CliffordBatchNorm1d is not None:
            self.clif_norm = CliffordBatchNorm1d(
                g=self.g,
                channels=self.C,
                eps=float(args.get("clifford_bn_eps", 1e-5)),
                momentum=float(args.get("clifford_bn_momentum", 0.1)),
                affine=True,
                track_running_stats=True,
            )
        else:
            self.clif_norm = nn.LayerNorm([self.C, self.n_blades])
        self.modrelu_bias = nn.Parameter(torch.zeros(self.C))

        # KvsAll attention scorer.
        self.q_proj = CliffordLinear(g=self.g, in_channels=self.C, out_channels=self.C, bias=True)
        self.k_proj = CliffordLinear(g=self.g, in_channels=self.C, out_channels=self.C, bias=True)
        self.v_proj = CliffordLinear(g=self.g, in_channels=self.C, out_channels=self.C, bias=True)
        self.fuse_proj = CliffordLinear(g=self.g, in_channels=2 * self.C, out_channels=self.C, bias=True)
        self.attn_dropout = nn.Dropout(float(args.get("attn_dropout", 0.1)))
        self.context_gate = nn.Parameter(torch.tensor(0.0))
        self.mv_act = MultiVectorAct(
            channels=self.C,
            algebra=IdentityBladeAlgebra(),
            input_blades=tuple(range(self.n_blades)),
            agg=args.get("mv_act_agg", "linear"),
        )

        # Choose which scorer forward_k_vs_all uses.
        self.use_kvsall_attention = bool(args.get("use_kvsall_attention", False))

    def hypercomplex_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return hypercomplex_multiply(a, b)

    def clifford_mul(self, a, b, g=None):
        """Convenience wrapper around generic Clifford multiplication helper."""
        if g is None:
            g = self.g
        return clifford_multiply(a, b, g)

    def modrelu(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        modReLU over blade dimension.
        x: (B, C, I)
        """
        mag = torch.linalg.norm(x, dim=-1)  # (B, C)
        scale = F.relu(mag + self.modrelu_bias.view(1, -1)) / (mag + eps)
        return x * scale.unsqueeze(-1)

    def k_vs_all_score(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.Tensor:
        """
        emb_h: (B, embedding_dim)
        emb_r: (B, embedding_dim)
        Returns:
            (B, num_entities)
        """
        h_hc = to_hypercomplex(emb_h, n_components=self.n_blades)
        r_hc = to_hypercomplex(emb_r, n_components=self.n_blades)

        x = torch.cat([h_hc, r_hc], dim=1)     # (B, 2C, I)
        skip = self.skip_proj(x)               # (B, C, I)
        z = self.clif_in(x)                     # (B, C, I)
        z = self.clif_norm(z)
        z = self.mv_act(z)
        z = self.clif_out(z)                    # (B, C, I)
        gate = torch.sigmoid(self.res_gate)
        z = z + gate * skip

        ent_hc = to_hypercomplex(self.entity_embeddings.weight, n_components=self.n_blades)
        scores = torch.einsum("bci,eci->be", z, ent_hc)
        return scores

    def k_vs_all_attention_score(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.Tensor:
        """
        True entity-attention KvsAll:
        1) Build Q from h,r.
        2) Attend over all entity keys K and values V.
        3) Fuse context with Q and rescore against K.
        """
        h_hc = to_hypercomplex(emb_h, n_components=self.n_blades)
        r_hc = to_hypercomplex(emb_r, n_components=self.n_blades)
        hr_hc = self.hypercomplex_mul(h_hc, r_hc)

        Q = self.q_proj(hr_hc)  # (B, C, I)

        ent_hc = to_hypercomplex(self.entity_embeddings.weight, n_components=self.n_blades)
        K_all = self.k_proj(ent_hc)  # (E, C, I)
        V_all = self.v_proj(ent_hc)  # (E, C, I)

        # Multi-head over blades: each blade is one head.
        # complex -> 2 heads, quaternion -> 4 heads.
        logits = torch.einsum("bci,eci->bie", Q, K_all)  # (B, n_blades, E)
        attn = torch.softmax(logits, dim=-1)

        attn = self.attn_dropout(attn)

        # Per-head (per-blade) context aggregation.
        context = torch.einsum("bie,eci->bci", attn, V_all)

        fused = torch.cat([Q, torch.sigmoid(self.context_gate) * context], dim=1)  # (B, 2C, I)
        Q2 = self.fuse_proj(fused)  # (B, C, I)

        scores = torch.einsum("bci,eci->be", Q2, K_all)
        scores = scores / math.sqrt(self.embedding_dim)
        return scores

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
        emb_head, emb_rel = self.get_head_relation_representation(x)
        if self.use_kvsall_attention:
            return self.k_vs_all_attention_score(emb_h=emb_head, emb_r=emb_rel)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        raise NotImplementedError("forward_k_vs_sample is not implemented in CLNN_KGE.")

    def score(self, h, r, t):
        raise NotImplementedError("score is not implemented in CLNN_KGE.")
