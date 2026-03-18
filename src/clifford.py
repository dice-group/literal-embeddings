import math

import torch
import torch.nn as nn
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear
from dicee.models.base_model import BaseKGE
from src.clifford_utils import (
    IdentityBladeAlgebra,
    MultiVectorAct,
    hypercomplex_multiply,
    to_hypercomplex,
)
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

    def _to_mv(self, x: torch.Tensor) -> torch.Tensor:
        return to_hypercomplex(x, n_components=self.n_blades)

    def _entity_mv(self) -> torch.Tensor:
        return self._to_mv(self.entity_embeddings.weight)

    def _query_encoder(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.Tensor:
        h_hc = to_hypercomplex(emb_h, n_components=self.n_blades)
        r_hc = to_hypercomplex(emb_r, n_components=self.n_blades)
        query = torch.cat([h_hc, r_hc], dim=1)
        skip = self.skip_proj(query)
        z = self.clif_in(query)
        z = self.clif_norm(z)
        z = self.mv_act(z)
        z = self.clif_out(z)
        return z + torch.sigmoid(self.res_gate) * skip

    def k_vs_all_score(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.Tensor:
        query = self._query_encoder(emb_h, emb_r)
        entity_mv = self._entity_mv()
        return torch.einsum("bci,eci->be", query, entity_mv)

    def k_vs_all_attention_score(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.Tensor:
        hr_hc = hypercomplex_multiply(self._to_mv(emb_h), self._to_mv(emb_r))
        query = self.q_proj(hr_hc)
        entity_mv = self._entity_mv()
        key = self.k_proj(entity_mv)
        value = self.v_proj(entity_mv)

        logits = torch.einsum("bci,eci->bie", query, key)
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_dropout(attn)
        context = torch.einsum("bie,eci->bci", attn, value)
        fused = torch.cat([query, torch.sigmoid(self.context_gate) * context], dim=1)
        refined_query = self.fuse_proj(fused)
        return torch.einsum("bci,eci->be", refined_query, key) / math.sqrt(self.embedding_dim)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
        emb_head, emb_rel = self.get_head_relation_representation(x)
        if self.use_kvsall_attention:
            return self.k_vs_all_attention_score(emb_h=emb_head, emb_r=emb_rel)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        raise NotImplementedError("forward_k_vs_sample is not implemented in CLNN_KGE.")

    def score(self, h, r, t):
        raise NotImplementedError("score is not implemented in CLNN_KGE.")
