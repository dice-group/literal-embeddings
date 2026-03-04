from dicee.models import (ComplEx, Keci, DistMult, OMult, QMult, TransE, DeCaL, DualE)
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd


class Gate(torch.nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 gate_activation=torch.sigmoid):

        super(Gate, self).__init__()
        self.output_size = output_size

        self.gate_activation = gate_activation
        self.g = torch.nn.Linear(input_size, output_size)
        self.g1 = torch.nn.Linear(output_size, output_size, bias=False)
        self.g2 = torch.nn.Linear(input_size-output_size, output_size, bias=False)
        self.gate_bias = torch.nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], 1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output

def load_num_lit(ent2idx, literal_dir, wrap=True):
    train_path = f"{literal_dir}/train.txt"
    df = pd.read_csv(train_path, header=None, sep="\t")
    # numeric_path = f"{literal_dir}/numerical_literals.txt"
    # if pd.io.common.file_exists(train_path):
    #     df = pd.read_csv(train_path, header=None, sep="\t")
    # else:
    #     df = pd.read_csv(numeric_path, header=None, sep="\t")
    rel2idx = {v: k for k, v in enumerate(df[1].unique())}
    numerical_literals = np.zeros([len(ent2idx), len(rel2idx)], dtype=np.float32)
    for i, (s, p, lit) in enumerate(df.values):
        try:
            numerical_literals[ent2idx[s], rel2idx[p]] = lit
        except KeyError:
            continue
    max_lit, min_lit = np.max(numerical_literals, axis=0), np.min(numerical_literals, axis=0)
    numerical_literals = (numerical_literals - min_lit) / (max_lit - min_lit + 1e-8)
    if wrap:
        return torch.autograd.Variable(torch.from_numpy(numerical_literals))
    return numerical_literals


class _LiteralEBase:
    literal_model_name = None

    def _init_literale(self, args, ent2idx=None, rel2idx=None):
        self.name = self.literal_model_name or f"{self.__class__.__name__}"
        self.literal_dir = args["dataset_dir"] + "/literals/"
        self.ent2idx = ent2idx
        self.rel2idx = rel2idx
        self.numerical_literals = load_num_lit(self.ent2idx, self.literal_dir, wrap=True)
        self.n_num_lit = self.numerical_literals.size(1)
        self.emb_num_lit = Gate(self.embedding_dim + self.n_num_lit, self.embedding_dim)

    def _literal_embeddings(self, x: torch.LongTensor):
        e_h = x[:, 0]
        numerical_literals = self.numerical_literals.to(x.device)
        emb_head, emb_rel = self.get_head_relation_representation(x)
        e1_lit = numerical_literals[e_h]
        emb_h = self.emb_num_lit(emb_head, e1_lit)
        emb_E = self.emb_num_lit(self.entity_embeddings.weight, numerical_literals)
        return emb_h, emb_rel, emb_E

class DistMult_LiteralE(_LiteralEBase, DistMult):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.literal_model_name = "DistMult_LiteralE"
        self._init_literale(args, ent2idx=ent2idx, rel2idx=rel2idx)


    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_h, emb_rel, emb_E = self._literal_embeddings(x)
        return super().k_vs_all_score(emb_h=emb_h, emb_r=emb_rel, emb_E=emb_E)
    

class ComplEx_LiteralE(_LiteralEBase, ComplEx):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.literal_model_name = "ComplEx_LiteralE"
        self._init_literale(args, ent2idx=ent2idx, rel2idx=rel2idx)


    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_h, emb_rel, emb_E = self._literal_embeddings(x)
        return super().k_vs_all_score(emb_h=emb_h, emb_r=emb_rel, emb_E=emb_E)


class Keci_LiteralE(_LiteralEBase, Keci):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.literal_model_name = "Keci_LiteralE"
        self._init_literale(args, ent2idx=ent2idx, rel2idx=rel2idx)

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_h, emb_rel, emb_E = self._literal_embeddings(x)
        return super().k_vs_all_score(emb_h, emb_rel, emb_E)


class OMult_LiteralE(_LiteralEBase, OMult):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.literal_model_name = "OMult_LiteralE"
        self._init_literale(args, ent2idx=ent2idx, rel2idx=rel2idx)

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_h, emb_rel, emb_E = self._literal_embeddings(x)
        return super().k_vs_all_score(emb_h, emb_rel, emb_E)


class QMult_LiteralE(_LiteralEBase, QMult):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.literal_model_name = "QMult_LiteralE"
        self._init_literale(args, ent2idx=ent2idx, rel2idx=rel2idx)

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_h, emb_rel, emb_E = self._literal_embeddings(x)
        return super().k_vs_all_score(emb_h, emb_rel, emb_E)


class TransE_LiteralE(_LiteralEBase, TransE):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.literal_model_name = "TransE_LiteralE"
        self._init_literale(args, ent2idx=ent2idx, rel2idx=rel2idx)

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_h, emb_rel, emb_E = self._literal_embeddings(x)
        distance = F.pairwise_distance(torch.unsqueeze(emb_h + emb_rel, 1), emb_E, p=self._norm)
        return self.margin - distance


class DeCaL_LiteralE(_LiteralEBase, DeCaL):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.literal_model_name = "DeCaL_LiteralE"
        self._init_literale(args, ent2idx=ent2idx, rel2idx=rel2idx)

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_h, emb_rel, emb_E = self._literal_embeddings(x)
        h0, hp, hq, hk = self.construct_cl_multivector(emb_h, re=self.re, p=self.p, q=self.q, r=self.r)
        r0, rp, rq, rk = self.construct_cl_multivector(emb_rel, re=self.re, p=self.p, q=self.q, r=self.r)

        h0, hp, hq, hk, h0, rp, rq, rk = self.apply_coefficients(h0, hp, hq, hk, h0, rp, rq, rk)
        t0 = emb_E[:, :self.re]
        h0r0t0 = torch.einsum("br,er->be", h0 * r0, t0)

        if self.p > 0:
            tp = emb_E[:, self.re: self.re + (self.re * self.p)].view(self.num_entities, self.re, self.p)
            hp_rp_t0 = torch.einsum("brp, er  -> be", hp * rp, t0)
            h0_rp_tp = torch.einsum("brp, erp -> be", torch.einsum("br,  brp -> brp", h0, rp), tp)
            hp_r0_tp = torch.einsum("brp, erp -> be", torch.einsum("brp, br  -> brp", hp, r0), tp)
            score_p = hp_rp_t0 + h0_rp_tp + hp_r0_tp
        else:
            score_p = 0

        if self.q > 0:
            num = self.re + (self.re * self.p)
            tq = emb_E[:, num:num + (self.re * self.q)].view(self.num_entities, self.re, self.q)
            h0_rq_tq = torch.einsum("brq, erq -> be", torch.einsum("br,  brq -> brq", h0, rq), tq)
            hq_r0_tq = torch.einsum("brq, erq -> be", torch.einsum("brq, br  -> brq", hq, r0), tq)
            hq_rq_t0 = torch.einsum("brq, er  -> be", hq * rq, t0)
            score_q = h0_rq_tq + hq_r0_tq - hq_rq_t0
        else:
            score_q = 0

        if self.r > 0:
            tk = emb_E[:, -(self.re * self.r):].view(self.num_entities, self.re, self.r)
            h0_rk_tk = torch.einsum("brk, erk -> be", torch.einsum("br,  brk -> brk", h0, rk), tk)
            hk_r0_tk = torch.einsum("brk, erk -> be", torch.einsum("brk, br  -> brk", hk, r0), tk)
            score_r = h0_rk_tk + hk_r0_tk
        else:
            score_r = 0

        if self.p >= 2:
            sigma_pp = torch.sum(self.compute_sigma_pp(hp, rp), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_pp = 0

        if self.q >= 2:
            sigma_qq = torch.sum(self.compute_sigma_qq(hq, rq), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_qq = 0

        if self.r >= 2:
            sigma_rr = torch.sum(self.compute_sigma_rr(hk, rk), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_rr = 0

        if self.p >= 2 and self.q >= 2:
            sigma_pq = torch.sum(self.compute_sigma_pq(hp=hp, hq=hq, rp=rp, rq=rq), dim=[1, 2, 3]).unsqueeze(-1)
        else:
            sigma_pq = 0
        if self.p >= 2 and self.r >= 2:
            sigma_pr = torch.sum(self.compute_sigma_pr(hp=hp, hk=hk, rp=rp, rk=rk), dim=[1, 2, 3]).unsqueeze(-1)
        else:
            sigma_pr = 0
        if self.q >= 2 and self.r >= 2:
            sigma_qr = torch.sum(self.compute_sigma_qr(hq=hq, hk=hk, rq=rq, rk=rk), dim=[1, 2, 3]).unsqueeze(-1)
        else:
            sigma_qr = 0

        return h0r0t0 + score_p + score_q + score_r + sigma_pp + sigma_qq + sigma_rr + sigma_pq + sigma_pr + sigma_qr


class DualE_LiteralE(_LiteralEBase, DualE):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.literal_model_name = "DualE_LiteralE"
        self._init_literale(args, ent2idx=ent2idx, rel2idx=rel2idx)

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_h, emb_rel, emb_E = self._literal_embeddings(x)

        e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h = torch.hsplit(emb_h, 8)
        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = torch.hsplit(emb_rel, 8)
        e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t = torch.hsplit(emb_E, 8)

        e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t = (
            self.T(e_1_t),
            self.T(e_2_t),
            self.T(e_3_t),
            self.T(e_4_t),
            self.T(e_5_t),
            self.T(e_6_t),
            self.T(e_7_t),
            self.T(e_8_t),
        )

        return self.kvsall_score(
            e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
            e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
            r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8,
        )
