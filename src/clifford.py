import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear
from cliffordlayers.nn.modules.cliffordconv import CliffordConv1d, CliffordConv2d, CliffordConv3d
from dicee.models.base_model import BaseKGE
try:
    from cliffordlayers.nn.modules.batchnorm import CliffordBatchNorm1d
except Exception:  # pragma: no cover
    CliffordBatchNorm1d = None


class MultiVectorAct(nn.Module):
    """
    Apply multivector activation with blade-wise gating.

    Args:
        channels: Number of feature channels.
        algebra: Algebra object exposing `embed(x, blades)` and `get(x, blades)`.
        input_blades: Blade indices present in the input.
        kernel_blades: Blade indices used to compute the gate. Defaults to `input_blades`.
        agg: Aggregation mode for gate computation: {"linear", "sum", "mean"}.
    """

    def __init__(self, channels, algebra, input_blades, kernel_blades=None, agg="linear"):
        super().__init__()
        self.algebra = algebra
        self.input_blades = tuple(input_blades)
        self.kernel_blades = tuple(kernel_blades) if kernel_blades is not None else self.input_blades
        self.agg = agg

        if self.agg == "linear":
            self.conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=len(self.kernel_blades),
                groups=channels,
            )
        elif self.agg not in {"sum", "mean"}:
            raise ValueError(f"Aggregation {self.agg} not implemented.")

    def forward(self, input):
        v = self.algebra.embed(input, self.input_blades)

        if self.agg == "linear":
            gate = torch.sigmoid(self.conv(v[..., self.kernel_blades]))
        elif self.agg == "sum":
            gate = torch.sigmoid(v[..., self.kernel_blades].sum(dim=-1, keepdim=True))
        elif self.agg == "mean":
            gate = torch.sigmoid(v[..., self.kernel_blades].mean(dim=-1, keepdim=True))
        else:
            raise ValueError(f"Aggregation {self.agg} not implemented.")

        v = v * gate
        v = self.algebra.get(v, self.input_blades)
        return v


class _IdentityBladeAlgebra:
    """Minimal adapter so MultiVectorAct can operate on already-embedded blade tensors."""

    @staticmethod
    def embed(x, blades):
        return x

    @staticmethod
    def get(x, blades):
        return x


@lru_cache(maxsize=32)
def _clifford_mul_table(g_tuple):
    """Build multiplication table for basis blades under signature g."""
    g = list(g_tuple)
    n = len(g)
    n_blades = 1 << n
    idx = torch.zeros((n_blades, n_blades), dtype=torch.long)
    sign = torch.zeros((n_blades, n_blades), dtype=torch.float32)

    for i in range(n_blades):
        for j in range(n_blades):
            s = 1.0
            for bit in range(n):
                if (i >> bit) & 1:
                    # Anti-commutation sign from swaps with lower-index basis vectors in j.
                    if (((j & ((1 << bit) - 1)).bit_count()) % 2) == 1:
                        s = -s
                    # Metric sign for repeated basis vector e_bit * e_bit = g[bit].
                    if ((j >> bit) & 1) == 1:
                        s = s * float(g[bit])
            idx[i, j] = i ^ j
            sign[i, j] = s
    return idx, sign


def clifford_multiply(a, b, g):
    """Generic Clifford product for scalars/multivectors under signature g.

    Args:
        a, b:
            Scalars or tensors whose last dim is n_blades (= 2**len(g)).
            Scalars are treated as pure scalar blades.
        g: Iterable signature, e.g. [-1] (complex-like), [-1, -1] (quaternion-like).

    Returns:
        Tensor of shape (..., n_blades) if multivector input is used, else scalar tensor.
    """
    g = tuple(float(x) for x in g)
    n_blades = 1 << len(g)

    a_t = torch.as_tensor(a)
    b_t = torch.as_tensor(b, device=a_t.device)
    dtype = torch.promote_types(a_t.dtype, b_t.dtype)
    if not dtype.is_floating_point:
        dtype = torch.float32
    a_t = a_t.to(dtype)
    b_t = b_t.to(dtype)

    a_scalar = a_t.ndim == 0
    b_scalar = b_t.ndim == 0

    # Lift scalars to pure-scalar multivectors.
    if a_scalar:
        a_mv = torch.zeros((n_blades,), dtype=dtype, device=a_t.device)
        a_mv[0] = a_t
    else:
        if a_t.shape[-1] == n_blades:
            a_mv = a_t
        elif a_t.shape[-1] == 1:
            a_mv = torch.zeros((*a_t.shape[:-1], n_blades), dtype=dtype, device=a_t.device)
            a_mv[..., 0] = a_t[..., 0]
        else:
            raise ValueError(f"a last dimension must be 1 or {n_blades}, got {a_t.shape[-1]}")

    if b_scalar:
        b_mv = torch.zeros((n_blades,), dtype=dtype, device=b_t.device)
        b_mv[0] = b_t
    else:
        if b_t.shape[-1] == n_blades:
            b_mv = b_t
        elif b_t.shape[-1] == 1:
            b_mv = torch.zeros((*b_t.shape[:-1], n_blades), dtype=dtype, device=b_t.device)
            b_mv[..., 0] = b_t[..., 0]
        else:
            raise ValueError(f"b last dimension must be 1 or {n_blades}, got {b_t.shape[-1]}")

    # Broadcast batch dimensions.
    a_mv, b_mv = torch.broadcast_tensors(a_mv, b_mv)
    out = torch.zeros_like(a_mv)

    idx, sign = _clifford_mul_table(g)
    idx = idx.to(device=a_mv.device)
    sign = sign.to(device=a_mv.device, dtype=dtype)

    for i in range(n_blades):
        ai = a_mv[..., i].unsqueeze(-1)  # (..., 1)
        for j in range(n_blades):
            out[..., idx[i, j]] += sign[i, j] * ai[..., 0] * b_mv[..., j]

    if a_scalar and b_scalar:
        return out[..., 0]
    return out


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
            algebra=_IdentityBladeAlgebra(),
            input_blades=tuple(range(self.n_blades)),
            agg=args.get("mv_act_agg", "linear"),
        )

        # Choose which scorer forward_k_vs_all uses.
        self.use_kvsall_attention = bool(args.get("use_kvsall_attention", False))

    @staticmethod
    def to_hypercomplex(x: torch.Tensor, n_components: int) -> torch.Tensor:
        """
        x: (..., n_components * d)
        returns: (..., d, n_components)
        """
        if x.shape[-1] % n_components != 0:
            raise ValueError(f"Last dimension ({x.shape[-1]}) not divisible by {n_components}")
        return x.reshape(*x.shape[:-1], n_components, -1).transpose(-2, -1)

    def hypercomplex_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a, b: (B, C, I) where I = n_blades
        returns: (B, C, I)
        """
        n_blades = a.size(-1)
        if n_blades == 2:
            return self._complex_mul(a, b)
        if n_blades == 4:
            return self._quaternion_mul(a, b)
        raise NotImplementedError(f"Hypercomplex multiplication not implemented for n_blades={n_blades}")

    def clifford_mul(self, a, b, g=None):
        """Convenience wrapper around generic Clifford multiplication helper."""
        if g is None:
            g = self.g
        return clifford_multiply(a, b, g)

    @staticmethod
    def _complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a0, a1 = a[..., 0], a[..., 1]
        b0, b1 = b[..., 0], b[..., 1]
        out0 = a0 * b0 - a1 * b1
        out1 = a0 * b1 + a1 * b0
        return torch.stack([out0, out1], dim=-1)

    @staticmethod
    def _quaternion_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

        out0 = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
        out1 = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2
        out2 = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1
        out3 = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0
        return torch.stack([out0, out1, out2, out3], dim=-1)

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
        h_hc = self.to_hypercomplex(emb_h, n_components=self.n_blades)
        r_hc = self.to_hypercomplex(emb_r, n_components=self.n_blades)

        x = torch.cat([h_hc, r_hc], dim=1)     # (B, 2C, I)
        skip = self.skip_proj(x)               # (B, C, I)
        z = self.clif_in(x)                     # (B, C, I)
        z = self.clif_norm(z)
        z = self.mv_act(z)
        z = self.clif_out(z)                    # (B, C, I)
        gate = torch.sigmoid(self.res_gate)
        z = z + gate * skip

        ent_hc = self.to_hypercomplex(self.entity_embeddings.weight, n_components=self.n_blades)
        scores = torch.einsum("bci,eci->be", z, ent_hc)
        return scores

    def k_vs_all_attention_score(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.Tensor:
        """
        True entity-attention KvsAll:
        1) Build Q from h,r.
        2) Attend over all entity keys K and values V.
        3) Fuse context with Q and rescore against K.
        """
        h_hc = self.to_hypercomplex(emb_h, n_components=self.n_blades)
        r_hc = self.to_hypercomplex(emb_r, n_components=self.n_blades)
        hr_hc = self.hypercomplex_mul(h_hc, r_hc)

        Q = self.q_proj(hr_hc)  # (B, C, I)

        ent_hc = self.to_hypercomplex(self.entity_embeddings.weight, n_components=self.n_blades)
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


class CliffConvKGE(BaseKGE):
    """Generic Clifford-convolution KGE model.

    A single model family that can emulate ConvE/ConvQ/ConvO-like behavior by
    selecting signature ``g``:
      - len(g)=1 (2 blades): complex-like
      - len(g)=2 (4 blades): quaternion-like
      - len(g)=3 (8 blades): octonion-like
    """

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

        # Factor C into H*W for 2D/3D conv view.
        self.h = int(args.get("cliff_conv_h", int(math.sqrt(self.C)) or 1))
        self.h = max(1, min(self.h, self.C))
        while self.C % self.h != 0 and self.h > 1:
            self.h -= 1
        self.w = self.C // self.h

        self.out_channels = int(args.get("num_of_output_channels", 8))
        k = int(args.get("kernel_size", 3))
        self.conv_dropout = nn.Dropout(float(args.get("feature_map_dropout_rate", 0.0)))
        self.hidden_dropout = nn.Dropout(float(args.get("hidden_dropout_rate", 0.0)))

        # Generic Clifford conv chosen by algebra dimension.
        if self.dim == 1:
            self.conv = CliffordConv1d(
                g=self.g,
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                bias=True,
            )
        elif self.dim == 2:
            self.conv = CliffordConv2d(
                g=self.g,
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                bias=True,
            )
        else:
            self.conv = CliffordConv3d(
                g=self.g,
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                bias=True,
            )
        self.conv_norm = nn.Identity()

        # Lazy init to avoid manual shape math for post-conv flatten dim.
        self.post_linear = None
        self.post_norm = None

    @staticmethod
    def to_hypercomplex(x: torch.Tensor, n_components: int) -> torch.Tensor:
        if x.shape[-1] % n_components != 0:
            raise ValueError(f"Last dimension ({x.shape[-1]}) not divisible by {n_components}")
        return x.reshape(*x.shape[:-1], n_components, -1).transpose(-2, -1)

    def _ensure_post_layers(self, conv_out: torch.Tensor):
        if self.post_linear is not None:
            return
        in_channels = int(conv_out.shape[1] * math.prod(conv_out.shape[2:-1]))
        self.post_linear = CliffordLinear(g=self.g, in_channels=in_channels, out_channels=self.C, bias=True).to(
            conv_out.device
        )
        self.post_norm = nn.LayerNorm([self.C, self.n_blades]).to(conv_out.device)

    def _encode_query(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.Tensor:
        h_hc = self.to_hypercomplex(emb_h, n_components=self.n_blades)  # (B, C, I)
        r_hc = self.to_hypercomplex(emb_r, n_components=self.n_blades)  # (B, C, I)

        if self.dim == 1:
            h_img = h_hc.reshape(h_hc.size(0), self.C, self.n_blades).unsqueeze(1)   # (B,1,C,I)
            r_img = r_hc.reshape(r_hc.size(0), self.C, self.n_blades).unsqueeze(1)   # (B,1,C,I)
            x = torch.cat([h_img, r_img], dim=2)                                       # (B,1,2C,I)
        elif self.dim == 2:
            h_img = h_hc.reshape(h_hc.size(0), self.h, self.w, self.n_blades).unsqueeze(1)  # (B,1,H,W,I)
            r_img = r_hc.reshape(r_hc.size(0), self.h, self.w, self.n_blades).unsqueeze(1)  # (B,1,H,W,I)
            x = torch.cat([h_img, r_img], dim=2)                                                # (B,1,2H,W,I)
        else:
            h_img = h_hc.reshape(h_hc.size(0), 1, self.h, self.w, self.n_blades).unsqueeze(1)  # (B,1,1,H,W,I)
            r_img = r_hc.reshape(r_hc.size(0), 1, self.h, self.w, self.n_blades).unsqueeze(1)  # (B,1,1,H,W,I)
            x = torch.cat([h_img, r_img], dim=2)                                                 # (B,1,2,H,W,I)

        z5 = self.conv(x)
        z5 = self.conv_norm(z5)
        z5 = self.conv_dropout(z5)
        self._ensure_post_layers(z5)
        z = z5.reshape(z5.size(0), -1, self.n_blades)  # (B, O*2H*W, I)
        q = self.post_linear(z)  # (B, C, I)
        q = self.post_norm(q)
        q = self.hidden_dropout(q)
        return q

    def k_vs_all_score(self, emb_h: torch.Tensor, emb_r: torch.Tensor) -> torch.Tensor:
        q = self._encode_query(emb_h, emb_r)  # (B, C, I)
        ent_hc = self.to_hypercomplex(self.entity_embeddings.weight, n_components=self.n_blades)  # (E, C, I)
        return torch.einsum("bci,eci->be", q, ent_hc)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
        emb_head, emb_rel = self.get_head_relation_representation(x)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        raise NotImplementedError("forward_k_vs_sample is not implemented in CliffConvKGE.")

    def score(self, h, r, t):
        q = self._encode_query(h, r)
        t_hc = self.to_hypercomplex(t, n_components=self.n_blades)
        return torch.einsum("bci,bci->b", q, t_hc)
