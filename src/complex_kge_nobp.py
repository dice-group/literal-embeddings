import torch
import torch.nn as nn
import torch.nn.functional as F
from dicee.models import BaseKGE
from torch.optim import Adam

from .layers import ComplexLinear


class ComplexKGENoBP(BaseKGE):
    """Complex FF-KGE with manual local updates (no autograd/backprop)."""

    def __init__(self, args):
        super().__init__(args)
        self.name = "ComplexKGENoBP"
        self.embedding_dim = args.get("embedding_dim", 32)
        self.in_dim = self.embedding_dim * 3
        self.hid_dim = self.in_dim

        self.ff_pos_weight = float(args.get("ff_pos_weight", 1.0))
        self.ff_neg_weight = float(args.get("ff_neg_weight", 1.0))
        self.hid_threshold = float(args.get("ff_hid_threshold", 1.0))
        self.out_threshold = float(args.get("ff_out_threshold", 1.0))
        self.emb_lr = float(args.get("no_bp_emb_lr", self.learning_rate))
        self.update_style = args.get("no_bp_update_style", "manual")
        self.local_epochs = int(args.get("no_bp_local_epochs", 1))
        self.max_layer_update_norm = float(args.get("no_bp_max_layer_update_norm", 1.0))
        self.max_emb_update_value = float(args.get("no_bp_max_emb_update_value", 0.1))
        self.eval_goodness_mode = args.get("no_bp_eval_goodness_mode", "out")

        # Keep bias off so manual update follows the clean quadratic-form derivation.
        self.in_layer = ComplexLinear(self.in_dim, self.hid_dim, bias=False)
        # Identity activation keeps manual local gradients analytically consistent.
        self.in_act = nn.Identity()
        self.out_layer = ComplexLinear(self.hid_dim, 1, bias=False)

        if self.update_style == "local_adam":
            self.in_opt = Adam(self.in_layer.parameters(), lr=self.learning_rate)
            self.out_opt = Adam(self.out_layer.parameters(), lr=self.learning_rate)
            self.emb_opt = Adam(
                [self.entity_embeddings.weight, self.relation_embeddings.weight],
                lr=self.emb_lr,
            )
        elif self.update_style != "manual":
            raise ValueError(
                f"Unknown no_bp_update_style={self.update_style}. Expected one of: manual, local_adam."
            )
        if self.eval_goodness_mode not in {"out", "sum"}:
            raise ValueError(
                f"Unknown no_bp_eval_goodness_mode={self.eval_goodness_mode}. Expected one of: out, sum."
            )

    @staticmethod
    def _to_complex(x: torch.Tensor) -> torch.Tensor:
        return torch.complex(x, torch.zeros_like(x))

    @staticmethod
    def _goodness(h: torch.Tensor) -> torch.Tensor:
        return (h.real.pow(2) + h.imag.pow(2)).mean(dim=-1)

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        # Same idea as the FF snippet: normalize each sample direction.
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        return x / (norm + eps)

    def _build_input_real(
        self, emb_h: torch.Tensor, emb_r: torch.Tensor, emb_t: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((emb_h, emb_r, emb_t), dim=-1)

    def get_input(self, x: torch.LongTensor) -> torch.Tensor:
        emb_h, emb_r, emb_t = self.get_triple_representation(x)
        return self._to_complex(self._build_input_real(emb_h, emb_r, emb_t))

    def _ff_goodness_loss(self, g_pos: torch.Tensor, g_neg: torch.Tensor, threshold: float):
        loss_pos = F.softplus(threshold - g_pos).mean()
        loss_neg = F.softplus(g_neg - threshold).mean()
        norm = max(self.ff_pos_weight + self.ff_neg_weight, 1e-8)
        loss = (self.ff_pos_weight * loss_pos + self.ff_neg_weight * loss_neg) / norm
        return loss, loss_pos, loss_neg

    def _ff_logistic_concat_loss(self, g_pos: torch.Tensor, g_neg: torch.Tensor, threshold: float):
        # Directly mirrors the update style from the provided FF snippet.
        logits = torch.cat((threshold - g_pos, g_neg - threshold), dim=0)
        # Numerically stable equivalent of log(1 + exp(.)).
        return F.softplus(logits).mean()

    @staticmethod
    def _local_grad_wrt_w_conj(
        x: torch.Tensor,
        y: torch.Tensor,
        coeff: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, I), y: (B, O), coeff: (B,)
        # grad sample: coeff_i * y_i x_i^H
        return (coeff[:, None, None] * y[:, :, None] * x.conj()[:, None, :]).sum(dim=0)

    def _manual_layer_update(
        self,
        layer: ComplexLinear,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        threshold: float,
        lr: float,
        activation=None,
    ):
        with torch.no_grad():
            y_pos_lin = layer(x_pos)
            y_neg_lin = layer(x_neg)
            y_pos = activation(y_pos_lin) if activation is not None else y_pos_lin
            y_neg = activation(y_neg_lin) if activation is not None else y_neg_lin
            y_pos = torch.nan_to_num(y_pos, nan=0.0, posinf=1e6, neginf=-1e6)
            y_neg = torch.nan_to_num(y_neg, nan=0.0, posinf=1e6, neginf=-1e6)

            g_pos = self._goodness(y_pos)
            g_neg = self._goodness(y_neg)
            g_pos = torch.nan_to_num(g_pos, nan=0.0, posinf=1e6, neginf=-1e6)
            g_neg = torch.nan_to_num(g_neg, nan=0.0, posinf=1e6, neginf=-1e6)
            loss, loss_pos, loss_neg = self._ff_goodness_loss(g_pos, g_neg, threshold)

            # d softplus(theta - g)/dg = -sigmoid(theta - g)
            # d softplus(g - theta)/dg = +sigmoid(g - theta)
            b_pos = max(int(g_pos.numel()), 1)
            b_neg = max(int(g_neg.numel()), 1)
            out_dim = max(int(y_pos.size(-1)), 1)
            norm = max(self.ff_pos_weight + self.ff_neg_weight, 1e-8)

            c_pos = (
                -(self.ff_pos_weight / norm)
                * torch.sigmoid(threshold - g_pos)
                / float(b_pos * out_dim)
            )
            c_neg = (
                (self.ff_neg_weight / norm)
                * torch.sigmoid(g_neg - threshold)
                / float(b_neg * out_dim)
            )

            grad_pos = self._local_grad_wrt_w_conj(x_pos, y_pos, c_pos)
            grad_neg = self._local_grad_wrt_w_conj(x_neg, y_neg, c_neg)
            grad = torch.nan_to_num(grad_pos + grad_neg, nan=0.0, posinf=1e6, neginf=-1e6)
            gnorm = torch.linalg.vector_norm(grad).item()
            if gnorm > self.max_layer_update_norm > 0.0:
                grad = grad * (self.max_layer_update_norm / (gnorm + 1e-12))
            layer.weight -= lr * grad
            layer.weight.copy_(torch.nan_to_num(layer.weight, nan=0.0, posinf=1e6, neginf=-1e6))

            return y_pos.detach(), y_neg.detach(), c_pos.detach(), c_neg.detach(), loss, loss_pos, loss_neg

    @staticmethod
    def _input_grad_from_local_coeff(
        layer: ComplexLinear,
        y: torch.Tensor,
        coeff: torch.Tensor,
    ) -> torch.Tensor:
        # For y = W x and g = mean(|y|^2), dx scales with W^H y.
        return coeff[:, None] * (y @ layer.weight.conj())

    def _manual_update_embeddings_from_input_grad(
        self,
        triples: torch.Tensor,
        dx: torch.Tensor,
        lr_emb: float,
    ):
        # Inputs are [h, r, t] concatenation in real space; embeddings are real-valued.
        dx_real = torch.nan_to_num(dx.real, nan=0.0, posinf=1e6, neginf=-1e6)
        if self.max_emb_update_value > 0.0:
            dx_real = dx_real.clamp(min=-self.max_emb_update_value, max=self.max_emb_update_value)
        d_h, d_r, d_t = torch.split(dx_real, self.embedding_dim, dim=-1)
        h_idx = triples[:, 0].long()
        r_idx = triples[:, 1].long()
        t_idx = triples[:, 2].long()

        self.entity_embeddings.weight.index_add_(0, h_idx, -lr_emb * d_h)
        self.relation_embeddings.weight.index_add_(0, r_idx, -lr_emb * d_r)
        self.entity_embeddings.weight.index_add_(0, t_idx, -lr_emb * d_t)
        self.entity_embeddings.weight.copy_(
            torch.nan_to_num(self.entity_embeddings.weight, nan=0.0, posinf=1e6, neginf=-1e6)
        )
        self.relation_embeddings.weight.copy_(
            torch.nan_to_num(self.relation_embeddings.weight, nan=0.0, posinf=1e6, neginf=-1e6)
        )

    def score(self, h, r, t):
        x = self._to_complex(self._build_input_real(h, r, t))
        hid = self.in_act(self.in_layer(x))
        out = self.out_layer(hid)
        g_out = self._goodness(out)
        if self.eval_goodness_mode == "sum":
            return self._goodness(hid) + g_out
        return g_out

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_h, emb_r = self.get_head_relation_representation(x)
        emb_t_all = self.entity_embeddings.weight

        bsz = emb_h.size(0)
        num_entities = emb_t_all.size(0)

        h_exp = emb_h.unsqueeze(1).expand(bsz, num_entities, -1)
        r_exp = emb_r.unsqueeze(1).expand(bsz, num_entities, -1)
        t_exp = emb_t_all.unsqueeze(0).expand(bsz, num_entities, -1)

        inp = self._build_input_real(h_exp, r_exp, t_exp).reshape(bsz * num_entities, -1)
        inp = self._to_complex(inp)

        hid = self.in_act(self.in_layer(inp))
        out = self.out_layer(hid)
        g_out = self._goodness(out)
        if self.eval_goodness_mode == "sum":
            score = self._goodness(hid) + g_out
        else:
            score = g_out
        return score.view(bsz, num_entities)

    def ff_update(self, x_pos, x_neg, optimizer=None):
        del optimizer  # no optimizer/backprop path by design
        lr = float(getattr(self, "learning_rate", 1e-3))

        if self.update_style == "local_adam":
            return self._ff_update_local_adam(x_pos, x_neg)

        x_pos_inp = self.get_input(x_pos)
        x_neg_inp = self.get_input(x_neg)

        hid_pos, hid_neg, c_pos, c_neg, hid_loss, hid_loss_pos, hid_loss_neg = self._manual_layer_update(
            layer=self.in_layer,
            x_pos=x_pos_inp,
            x_neg=x_neg_inp,
            threshold=self.hid_threshold,
            lr=lr,
            activation=self.in_act,
        )
        with torch.no_grad():
            dx_pos = self._input_grad_from_local_coeff(self.in_layer, hid_pos, c_pos)
            dx_neg = self._input_grad_from_local_coeff(self.in_layer, hid_neg, c_neg)
            self._manual_update_embeddings_from_input_grad(x_pos, dx_pos, self.emb_lr)
            self._manual_update_embeddings_from_input_grad(x_neg, dx_neg, self.emb_lr)

        # Use hidden activations from updated layer-1 for local layer-2 update.
        with torch.no_grad():
            hid_pos = self.in_act(self.in_layer(x_pos_inp))
            hid_neg = self.in_act(self.in_layer(x_neg_inp))

        _, _, _, _, out_loss, out_loss_pos, out_loss_neg = self._manual_layer_update(
            layer=self.out_layer,
            x_pos=hid_pos,
            x_neg=hid_neg,
            threshold=self.out_threshold,
            lr=lr,
        )

        total_loss = hid_loss.detach() + out_loss.detach()
        return {
            "loss": float(total_loss.item()),
            "hid_loss": hid_loss.item(),
            "out_loss": out_loss.item(),
            "hid_pos_loss": hid_loss_pos.item(),
            "hid_neg_loss": hid_loss_neg.item(),
            "out_pos_loss": out_loss_pos.item(),
            "out_neg_loss": out_loss_neg.item(),
        }

    def _ff_update_local_adam(self, x_pos, x_neg):
        hid_loss = torch.tensor(0.0, device=x_pos.device)
        out_loss = torch.tensor(0.0, device=x_pos.device)
        hid_pos_loss = torch.tensor(0.0, device=x_pos.device)
        hid_neg_loss = torch.tensor(0.0, device=x_pos.device)
        out_pos_loss = torch.tensor(0.0, device=x_pos.device)
        out_neg_loss = torch.tensor(0.0, device=x_pos.device)

        for _ in range(max(self.local_epochs, 1)):
            # Rebuild inputs each local epoch so embedding updates are reflected.
            x_pos_inp = self.get_input(x_pos)
            x_neg_inp = self.get_input(x_neg)

            # Layer-1 local step (+ embedding update), similar to provided FF style.
            h_pos = self.in_act(self.in_layer(self._normalize_input(x_pos_inp)))
            h_neg = self.in_act(self.in_layer(self._normalize_input(x_neg_inp)))
            g_pos_h = self._goodness(h_pos)
            g_neg_h = self._goodness(h_neg)
            hid_loss = self._ff_logistic_concat_loss(g_pos_h, g_neg_h, self.hid_threshold)
            hid_pos_loss = F.softplus(self.hid_threshold - g_pos_h).mean()
            hid_neg_loss = F.softplus(g_neg_h - self.hid_threshold).mean()

            self.in_opt.zero_grad()
            self.emb_opt.zero_grad()
            hid_loss.backward()
            self.in_opt.step()
            self.emb_opt.step()

            # Layer-2 local step (detach hidden states from layer-1).
            with torch.no_grad():
                h_pos_d = self.in_act(self.in_layer(self._normalize_input(x_pos_inp))).detach()
                h_neg_d = self.in_act(self.in_layer(self._normalize_input(x_neg_inp))).detach()
            z_pos = self.out_layer(self._normalize_input(h_pos_d))
            z_neg = self.out_layer(self._normalize_input(h_neg_d))
            g_pos_o = self._goodness(z_pos)
            g_neg_o = self._goodness(z_neg)
            out_loss = self._ff_logistic_concat_loss(g_pos_o, g_neg_o, self.out_threshold)
            out_pos_loss = F.softplus(self.out_threshold - g_pos_o).mean()
            out_neg_loss = F.softplus(g_neg_o - self.out_threshold).mean()

            self.out_opt.zero_grad()
            out_loss.backward()
            self.out_opt.step()

        total_loss = hid_loss.detach() + out_loss.detach()
        return {
            "loss": float(total_loss.item()),
            "hid_loss": hid_loss.item(),
            "out_loss": out_loss.item(),
            "hid_pos_loss": hid_pos_loss.item(),
            "hid_neg_loss": hid_neg_loss.item(),
            "out_pos_loss": out_pos_loss.item(),
            "out_neg_loss": out_neg_loss.item(),
        }
