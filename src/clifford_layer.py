import torch
import torch.nn as nn


class CliffordSignature:
    """Project-local Clifford signature helper.

    Supports `g=[]` (0D signature) and any list/tuple/tensor metric.
    """

    def __init__(self, g):
        self.g = self._as_tensor(g)
        self.dim = int(self.g.numel())
        self.n_blades = int(2 ** self.dim)

    @staticmethod
    def _as_tensor(g) -> torch.Tensor:
        if g is None:
            return torch.empty(0, dtype=torch.float32)
        if isinstance(g, torch.Tensor):
            return g.to(dtype=torch.float32).flatten()
        if isinstance(g, (list, tuple)):
            return torch.as_tensor(g, dtype=torch.float32).flatten()
        raise ValueError("Unknown signature type. Expected list/tuple/tensor/None.")


class CliffordLinear(nn.Module):
    """Project-local Clifford linear layer.

    This implementation applies a learnable linear map per blade:
      x: (B, in_channels, n_blades) -> y: (B, out_channels, n_blades)
    """

    def __init__(self, g, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        sig = CliffordSignature(g)
        self.g = sig.g
        self.dim = sig.dim
        self.n_blades = sig.n_blades
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.empty(self.n_blades, out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.n_blades, out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1.0 / max(self.in_channels, 1) ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B,C,I), got shape={tuple(x.shape)}")
        if x.shape[-1] != self.n_blades:
            raise ValueError(
                f"Input has {x.shape[-1]} blades, but layer expects {self.n_blades}."
            )
        out = torch.einsum("bci,ioc->boi", x, self.weight)
        if self.bias is not None:
            out = out + self.bias.transpose(0, 1).unsqueeze(0)
        return out
