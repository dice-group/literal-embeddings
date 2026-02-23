import torch
import torch.nn as nn
import torch.nn.functional as F
from dicee.models import BaseKGE

from .layers import ComplexLinear, FFLayer, FFOutLayer, ModReLUComplex


class CLNN(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = "CLNN"

        self.g = args.get("clifford_g", [-1])
        self.n_blades = 2 ** (len(self.g))
        self.embedding_dim = args.get('embedding_dim', 32)
        self.in_channels = self.embedding_dim * 3 // self.n_blades
        self.hid_channels = self.embedding_dim * 3 // self.n_blades
        self.out_channels = 1
        self.ff_pos_weight = float(args.get("ff_pos_weight", 1.0))
        self.ff_neg_weight = float(args.get("ff_neg_weight", 1.0))
        self.ff_out_agg = args.get("ff_out_agg", "mean")

        self.in_layer = FFLayer(
            g=self.g,
            in_channels=self.in_channels,
            out_channels=self.hid_channels,
            lr=self.learning_rate,
            pos_weight=self.ff_pos_weight,
            neg_weight=self.ff_neg_weight,
        )
        self.out_layer = FFOutLayer(
            g=self.g,
            in_channels=self.hid_channels,
            out_channels=self.out_channels,
            lr=self.learning_rate,
            pos_weight=self.ff_pos_weight,
            neg_weight=self.ff_neg_weight,
            agg_mode=self.ff_out_agg,
        )
        
    @staticmethod
    def to_hypercomplex(x: torch.Tensor, n_components: int) -> torch.Tensor:
        """
        x: (..., n_components * d)
        returns: (..., d, n_components)
        """
        if x.shape[-1] % n_components != 0:
            raise ValueError(f"Last dimension ({x.shape[-1]}) not divisible by {n_components}")
        return x.reshape(*x.shape[:-1], n_components, -1).transpose(-2, -1)
    

    def get_input(self, x):
        emb_head, emb_rel, emb_tail = self.get_triple_representation(x)
        h_hc = self.to_hypercomplex(emb_head, n_components=self.n_blades)
        r_hc = self.to_hypercomplex(emb_rel, n_components=self.n_blades)
        t_hc = self.to_hypercomplex(emb_tail, n_components=self.n_blades)
        return torch.cat((h_hc, r_hc, t_hc), dim=1)

    def score(self, h,r,t):
        h_hc = self.to_hypercomplex(h, n_components=self.n_blades)
        r_hc = self.to_hypercomplex(r, n_components=self.n_blades)
        t_hc = self.to_hypercomplex(t, n_components=self.n_blades)
        hc_inp = torch.cat((h_hc, r_hc, t_hc), dim=1)
        hid = self.in_layer.forward(hc_inp)
        out = self.out_layer.forward(hid)
        return out

    def forward_k_vs_all(self, x: torch.LongTensor):
        """Scores all candidate tails for each (head, relation) pair.

        x: (B, 2) where columns are (h, r)
        returns: (B, num_entities)
        """
        emb_head, emb_rel = self.get_head_relation_representation(x)
        emb_tail_all = self.entity_embeddings.weight  # (E, D)

        h_hc = self.to_hypercomplex(emb_head, n_components=self.n_blades)      # (B, C, I)
        r_hc = self.to_hypercomplex(emb_rel, n_components=self.n_blades)       # (B, C, I)
        t_hc_all = self.to_hypercomplex(emb_tail_all, n_components=self.n_blades)  # (E, C, I)

        bsz = h_hc.size(0)
        num_entities = t_hc_all.size(0)

        h_exp = h_hc.unsqueeze(1).expand(bsz, num_entities, -1, -1)
        r_exp = r_hc.unsqueeze(1).expand(bsz, num_entities, -1, -1)
        t_exp = t_hc_all.unsqueeze(0).expand(bsz, num_entities, -1, -1)

        # Merge (B, E) to run the Clifford layers in a single batched call.
        hc_inp = torch.cat((h_exp, r_exp, t_exp), dim=2).reshape(
            bsz * num_entities, -1, self.n_blades
        )
        hid = self.in_layer.forward(hc_inp)
        out = self.out_layer.forward(hid)  # (B*E,)
        return out.view(bsz, num_entities)

    def ff_update(self, x_pos, x_neg, optimizer):
        x_pos_inp = self.get_input(x_pos)
        x_neg_inp = self.get_input(x_neg)
        hid_pos, hid_neg, hid_loss, hid_loss_pos, hid_loss_neg = self.in_layer.train_step(x_pos_inp, x_neg_inp)

        optimizer.zero_grad()
        hid_loss.backward()
        optimizer.step()

        with torch.no_grad():
            hid_pos = self.in_layer.forward(x_pos_inp)
            hid_neg = self.in_layer.forward(x_neg_inp)

        out_loss, out_loss_pos, out_loss_neg = self.out_layer.train_step(hid_pos.detach(), hid_neg.detach())
        optimizer.zero_grad()
        out_loss.backward()
        optimizer.step()

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


class ComplexKGE(BaseKGE):
    """Complex FF-KGE model with quadratic-form goodness objectives."""

    def __init__(self, args):
        super().__init__(args)
        self.name = "ComplexKGE"
        self.embedding_dim = args.get("embedding_dim", 32)
        self.in_dim = self.embedding_dim * 3
        self.hid_dim = self.in_dim
        self.ff_pos_weight = float(args.get("ff_pos_weight", 1.0))
        self.ff_neg_weight = float(args.get("ff_neg_weight", 1.0))

        self.hid_threshold = float(args.get("ff_hid_threshold", 1.0))
        self.out_threshold = float(args.get("ff_out_threshold", 1.0))

        self.in_layer = ComplexLinear(self.in_dim, self.hid_dim)
        self.in_act = ModReLUComplex(self.hid_dim)
        self.out_layer = ComplexLinear(self.hid_dim, 1)

    @staticmethod
    def _to_complex(x: torch.Tensor) -> torch.Tensor:
        return torch.complex(x, torch.zeros_like(x))

    def _build_input_real(
        self, emb_h: torch.Tensor, emb_r: torch.Tensor, emb_t: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((emb_h, emb_r, emb_t), dim=-1)

    def get_input(self, x: torch.LongTensor) -> torch.Tensor:
        emb_h, emb_r, emb_t = self.get_triple_representation(x)
        inp_real = self._build_input_real(emb_h, emb_r, emb_t)
        return self._to_complex(inp_real)

    @staticmethod
    def _goodness(h: torch.Tensor) -> torch.Tensor:
        return (h.real.pow(2) + h.imag.pow(2)).mean(dim=-1)

    def _ff_goodness_loss(self, g_pos: torch.Tensor, g_neg: torch.Tensor, threshold: float):
        loss_pos = F.softplus(threshold - g_pos).mean()
        loss_neg = F.softplus(g_neg - threshold).mean()
        norm = max(self.ff_pos_weight + self.ff_neg_weight, 1e-8)
        loss = (self.ff_pos_weight * loss_pos + self.ff_neg_weight * loss_neg) / norm
        return loss, loss_pos, loss_neg

    def _hid_train_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor):
        h_pos = self.in_act(self.in_layer(x_pos))
        h_neg = self.in_act(self.in_layer(x_neg))
        g_pos = self._goodness(h_pos)
        g_neg = self._goodness(h_neg)
        loss, loss_pos, loss_neg = self._ff_goodness_loss(
            g_pos=g_pos, g_neg=g_neg, threshold=self.hid_threshold
        )
        return h_pos, h_neg, loss, loss_pos, loss_neg

    def _out_train_step(self, h_pos: torch.Tensor, h_neg: torch.Tensor):
        z_pos = self.out_layer(h_pos)
        z_neg = self.out_layer(h_neg)
        g_pos = self._goodness(z_pos)
        g_neg = self._goodness(z_neg)
        loss, loss_pos, loss_neg = self._ff_goodness_loss(
            g_pos=g_pos, g_neg=g_neg, threshold=self.out_threshold
        )
        return loss, loss_pos, loss_neg

    def score(self, h, r, t):
        x = self._to_complex(self._build_input_real(h, r, t))
        hid = self.in_act(self.in_layer(x))
        out = self.out_layer(hid)
        return self._goodness(out)

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_h, emb_r = self.get_head_relation_representation(x)
        emb_t_all = self.entity_embeddings.weight
        bsz = emb_h.size(0)
        num_entities = emb_t_all.size(0)
        h_exp = emb_h.unsqueeze(1).expand(bsz, num_entities, -1)
        r_exp = emb_r.unsqueeze(1).expand(bsz, num_entities, -1)
        t_exp = emb_t_all.unsqueeze(0).expand(bsz, num_entities, -1)
        inp = self._build_input_real(h_exp, r_exp, t_exp).reshape(
            bsz * num_entities, -1
        )
        hid = self.in_act(self.in_layer(self._to_complex(inp)))
        out = self.out_layer(hid)
        score = self._goodness(out)
        return score.view(bsz, num_entities)

    def ff_update(self, x_pos, x_neg, optimizer):
        x_pos_inp = self.get_input(x_pos)
        x_neg_inp = self.get_input(x_neg)
        hid_pos, hid_neg, hid_loss, hid_loss_pos, hid_loss_neg = self._hid_train_step(x_pos_inp, x_neg_inp)

        optimizer.zero_grad()
        hid_loss.backward()
        optimizer.step()

        with torch.no_grad():
            hid_pos = self.in_act(self.in_layer(x_pos_inp))
            hid_neg = self.in_act(self.in_layer(x_neg_inp))

        out_loss, out_loss_pos, out_loss_neg = self._out_train_step(hid_pos.detach(), hid_neg.detach())
        optimizer.zero_grad()
        out_loss.backward()
        optimizer.step()

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
