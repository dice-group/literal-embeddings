import math

import torch
import torch.nn as nn
from cliffordlayers.nn.modules.cliffordconv import (
    CliffordConv1d,
    CliffordConv2d,
    CliffordConv3d,
)
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear
from dicee.models.base_model import BaseKGE
from src.clifford_utils import to_hypercomplex


class CliffConvKGE(BaseKGE):
    """Generic Clifford-convolution KGE model."""

    def __init__(self, args, edge_index=None, edge_type=None, entity2idx=None):
        super().__init__(args)
        self.name = "CliffConvKGE"
        self.edge_index = edge_index
        self.edge_type = edge_type

        g_from_args = args.get("clifford_g", None)
        if g_from_args is not None:
            self.g = list(g_from_args)
        else:
            p = int(args.get("p", 0))
            q = int(args.get("q", 1))
            self.g = ([1.0] * p) + ([-1.0] * q)

        self.n_blades = 2 ** len(self.g)
        self.dim = len(self.g)
        if self.embedding_dim % self.n_blades != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by n_blades ({self.n_blades})."
            )
        if self.dim not in {1, 2, 3}:
            raise ValueError(
                f"CliffConvKGE supports len(g) in {{1,2,3}} for Conv1d/2d/3d, got len(g)={self.dim}."
            )
        self.C = self.embedding_dim // self.n_blades

        self.h = int(args.get("cliff_conv_h", int(math.sqrt(self.C)) or 1))
        self.h = max(1, min(self.h, self.C))
        while self.C % self.h != 0 and self.h > 1:
            self.h -= 1
        self.w = self.C // self.h

        self.out_channels = int(args.get("num_of_output_channels", 8))
        kernel_size = int(args.get("kernel_size", 3))
        self.conv_dropout = nn.Dropout(float(args.get("feature_map_dropout_rate", 0.0)))
        self.hidden_dropout = nn.Dropout(float(args.get("hidden_dropout_rate", 0.0)))

        if self.dim == 1:
            self.conv = CliffordConv1d(
                g=self.g,
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True,
            )
        elif self.dim == 2:
            self.conv = CliffordConv2d(
                g=self.g,
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True,
            )
        else:
            self.conv = CliffordConv3d(
                g=self.g,
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True,
            )
        self.conv_norm = nn.Identity()
        self.post_linear = None
        self.post_norm = None

    def _ensure_post_layers(self, conv_out: torch.Tensor):
        if self.post_linear is not None:
            return
        in_channels = int(conv_out.shape[1] * math.prod(conv_out.shape[2:-1]))
        self.post_linear = CliffordLinear(
            g=self.g,
            in_channels=in_channels,
            out_channels=self.C,
            bias=True,
        ).to(conv_out.device)
        self.post_norm = nn.LayerNorm([self.C, self.n_blades]).to(conv_out.device)

    def _encode_query(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.Tensor:
        h_hc = to_hypercomplex(emb_h, n_components=self.n_blades)
        r_hc = to_hypercomplex(emb_r, n_components=self.n_blades)

        if self.dim == 1:
            h_img = h_hc.reshape(h_hc.size(0), self.C, self.n_blades).unsqueeze(1)
            r_img = r_hc.reshape(r_hc.size(0), self.C, self.n_blades).unsqueeze(1)
            x = torch.cat([h_img, r_img], dim=2)
        elif self.dim == 2:
            h_img = h_hc.reshape(h_hc.size(0), self.h, self.w, self.n_blades).unsqueeze(1)
            r_img = r_hc.reshape(r_hc.size(0), self.h, self.w, self.n_blades).unsqueeze(1)
            x = torch.cat([h_img, r_img], dim=2)
        else:
            h_img = h_hc.reshape(h_hc.size(0), 1, self.h, self.w, self.n_blades).unsqueeze(1)
            r_img = r_hc.reshape(r_hc.size(0), 1, self.h, self.w, self.n_blades).unsqueeze(1)
            x = torch.cat([h_img, r_img], dim=2)

        z5 = self.conv(x)
        z5 = self.conv_norm(z5)
        z5 = self.conv_dropout(z5)
        self._ensure_post_layers(z5)
        z = z5.reshape(z5.size(0), -1, self.n_blades)
        q = self.post_linear(z)
        q = self.post_norm(q)
        q = self.hidden_dropout(q)
        return q

    def k_vs_all_score(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.Tensor:
        q = self._encode_query(emb_h, emb_r)
        ent_hc = to_hypercomplex(
            self.entity_embeddings.weight, n_components=self.n_blades
        )
        return torch.einsum("bci,eci->be", q, ent_hc)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
        emb_head, emb_rel = self.get_head_relation_representation(x)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        raise NotImplementedError("forward_k_vs_sample is not implemented in CliffConvKGE.")

    def score(self, h, r, t):
        q = self._encode_query(h, r)
        t_hc = to_hypercomplex(t, n_components=self.n_blades)
        return torch.einsum("bci,bci->b", q, t_hc)
