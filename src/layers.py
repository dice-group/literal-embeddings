import torch
import torch.nn as nn
import torch.nn.functional as F
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear


class ComplexLinear(nn.Module):
    """Complex-valued linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.in_features
        fan_out = self.out_features
        bound = (6.0 / max(fan_in + fan_out, 1)) ** 0.5
        with torch.no_grad():
            real = torch.empty_like(self.weight.real).uniform_(-bound, bound)
            imag = torch.empty_like(self.weight.imag).uniform_(-bound, bound)
            self.weight.copy_(torch.complex(real, imag))
            if self.bias is not None:
                b_real = torch.empty_like(self.bias.real).uniform_(-bound, bound)
                b_imag = torch.empty_like(self.bias.imag).uniform_(-bound, bound)
                self.bias.copy_(torch.complex(b_real, b_imag))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ModReLUComplex(nn.Module):
    """modReLU activation for complex features."""

    def __init__(self, features: int, eps: float = 1e-8):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(features, dtype=torch.float32))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r = torch.abs(z)
        scale = F.relu(r + self.b) / (r + self.eps)
        return z * scale


class FFLayer(nn.Module):
    def __init__(self, g,  in_channels : int, out_channels : int,
                   lr :int, threshold : int = 2.0,   bias : bool = True,
                   pos_weight: float = 1.0, neg_weight: float = 1.0):
        super().__init__()
        self.g = g
        self.bias = bias
        self.lr = lr 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        
        self.threshold = threshold
        self.layer = CliffordLinear(g = self.g, in_channels= self.in_channels,
                                      out_channels= self.out_channels)
        self.modrelu_bias = nn.Parameter(torch.zeros(self.out_channels))
    
    def modrelu(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        modReLU over blade dimension.
        x: (B, C, I)
        """
        mag = torch.linalg.norm(x, dim=-1)  # (B, C)
        scale = F.relu(mag + self.modrelu_bias.view(1, -1)) / (mag + eps)
        return x * scale.unsqueeze(-1)

    def forward(self, x):
        return self.modrelu(self.layer(x))

    def goodness(self, activations):
        return (activations**2).mean(dim=1)

    def train_step(self, x_pos, x_neg):
        act_pos = self.forward(x_pos)
        act_neg = self.forward(x_neg)

        g_pos = self.goodness(act_pos)
        g_neg = self.goodness(act_neg)

        loss_pos = F.softplus(self.threshold - g_pos).mean()
        loss_neg = F.softplus(g_neg - self.threshold).mean()
        norm = max(self.pos_weight + self.neg_weight, 1e-8)
        loss = (self.pos_weight * loss_pos + self.neg_weight * loss_neg) / norm
        return act_pos, act_neg, loss, loss_pos, loss_neg
    
class FFOutLayer(nn.Module):
    def __init__(self, g,  in_channels : int, out_channels : int,
                 lr :int,threshold : int = 3.0 ,    bias : bool = True,
                 pos_weight: float = 1.0, neg_weight: float = 1.0,
                 agg_mode: str = "mean"):
        super().__init__()
        self.g = g
        self.bias = bias
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.agg_mode = agg_mode
        self.n_blades = 2 ** len(self.g)
        
        self.threshold = threshold
        self.layer = CliffordLinear(g = self.g, in_channels= self.in_channels,
                                      out_channels= self.out_channels)
        if self.agg_mode == "learned":
            self.blade_agg_logits = nn.Parameter(torch.zeros(self.n_blades))
        else:
            self.register_parameter("blade_agg_logits", None)
    
    def forward(self, x):
        blade_logits = self.layer(x)[:, 0]  # (batch, n_blades)
        if self.agg_mode == "blade0":
            return blade_logits[:, 0]
        if self.agg_mode == "mean":
            return blade_logits.mean(dim=-1)
        if self.agg_mode == "max":
            return blade_logits.max(dim=-1).values
        if self.agg_mode == "learned":
            weights = torch.softmax(self.blade_agg_logits, dim=0)  # (n_blades,)
            return (blade_logits * weights.view(1, -1)).sum(dim=-1)
        raise ValueError(
            f"Unknown agg_mode={self.agg_mode}. Expected one of: blade0, mean, max, learned."
        )

    def goodness(self, score):
        """Goodness for last layer = directly the scalar output"""
        return score  # (batch,)

    def train_step(self, x_pos, x_neg):
        act_pos = self.forward(x_pos)
        act_neg = self.forward(x_neg)

        g_pos = self.goodness(act_pos) - self.threshold
        g_neg = self.goodness(act_neg) - self.threshold

        targets_pos = torch.ones_like(g_pos)
        targets_neg = torch.zeros_like(g_neg)

        loss_pos = F.binary_cross_entropy_with_logits(g_pos, targets_pos)
        loss_neg = F.binary_cross_entropy_with_logits(g_neg, targets_neg)
        norm = max(self.pos_weight + self.neg_weight, 1e-8)
        loss = (self.pos_weight * loss_pos + self.neg_weight * loss_neg) / norm
        return loss, loss_pos, loss_neg
