from functools import lru_cache

import torch
import torch.nn as nn


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
        self.kernel_blades = (
            tuple(kernel_blades) if kernel_blades is not None else self.input_blades
        )
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


class IdentityBladeAlgebra:
    """Minimal adapter so MultiVectorAct can operate on already-embedded blade tensors."""

    @staticmethod
    def embed(x, blades):
        return x

    @staticmethod
    def get(x, blades):
        return x


def to_hypercomplex(x: torch.Tensor, n_components: int) -> torch.Tensor:
    if x.shape[-1] % n_components != 0:
        raise ValueError(
            f"Last dimension ({x.shape[-1]}) not divisible by {n_components}"
        )
    return x.reshape(*x.shape[:-1], n_components, -1).transpose(-2, -1)


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
                    if (((j & ((1 << bit) - 1)).bit_count()) % 2) == 1:
                        s = -s
                    if ((j >> bit) & 1) == 1:
                        s = s * float(g[bit])
            idx[i, j] = i ^ j
            sign[i, j] = s
    return idx, sign


def clifford_multiply(a, b, g):
    """Generic Clifford product for scalars/multivectors under signature g."""
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

    a_mv, b_mv = torch.broadcast_tensors(a_mv, b_mv)
    out = torch.zeros_like(a_mv)

    idx, sign = _clifford_mul_table(g)
    idx = idx.to(device=a_mv.device)
    sign = sign.to(device=a_mv.device, dtype=dtype)

    for i in range(n_blades):
        ai = a_mv[..., i].unsqueeze(-1)
        for j in range(n_blades):
            out[..., idx[i, j]] += sign[i, j] * ai[..., 0] * b_mv[..., j]

    if a_scalar and b_scalar:
        return out[..., 0]
    return out


def complex_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a0, a1 = a[..., 0], a[..., 1]
    b0, b1 = b[..., 0], b[..., 1]
    out0 = a0 * b0 - a1 * b1
    out1 = a0 * b1 + a1 * b0
    return torch.stack([out0, out1], dim=-1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    out0 = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
    out1 = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2
    out2 = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1
    out3 = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0
    return torch.stack([out0, out1, out2, out3], dim=-1)


def hypercomplex_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n_blades = a.size(-1)
    if n_blades == 2:
        return complex_multiply(a, b)
    if n_blades == 4:
        return quaternion_multiply(a, b)
    raise NotImplementedError(
        f"Hypercomplex multiplication not implemented for n_blades={n_blades}"
    )
