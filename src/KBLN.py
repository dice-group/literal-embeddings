from dicee.models import (ComplEx, Keci, DistMult, OMult, QMult, TransE, DeCaL, DualE)
import torch
import numpy as np
import pandas as pd


def load_num_lit(ent2idx, literal_dir, wrap=True):
    train_path = f"{literal_dir}/train.txt"
    numeric_path = f"{literal_dir}/numerical_literals.txt"
    if pd.io.common.file_exists(train_path):
        df = pd.read_csv(train_path, header=None, sep="\t")
    else:
        df = pd.read_csv(numeric_path, header=None, sep="\t")
    attr2idx = {v: k for k, v in enumerate(df[1].unique())}
    numerical_literals = np.zeros([len(ent2idx), len(attr2idx)], dtype=np.float32)
    for s, p, lit in df.values:
        try:
            numerical_literals[ent2idx[s], attr2idx[p]] = lit
        except KeyError:
            continue
    max_lit, min_lit = np.max(numerical_literals, axis=0), np.min(numerical_literals, axis=0)
    numerical_literals = (numerical_literals - min_lit) / (max_lit - min_lit + 1e-8)
    if wrap:
        return torch.autograd.Variable(torch.from_numpy(numerical_literals))
    return numerical_literals


def load_num_lit_kbln(ent2idx, rel2idx, dataset_dir):
    def load_rel_data(file_path, ent2idx, rel2idx):
        df = pd.read_csv(file_path, sep="\t", header=None)
        triples = np.zeros([df.shape[0], 3], dtype=np.int32)
        for i, row in df.iterrows():
            try:
                triples[i, 0] = ent2idx[row[0]]
                triples[i, 1] = rel2idx[row[1]]
                triples[i, 2] = ent2idx[row[2]]
            except KeyError:
                continue
        return triples

    literal_dir = f"{dataset_dir}/literals"
    numerical_literals = load_num_lit(ent2idx, literal_dir, wrap=False)
    train_triples = load_rel_data(f"{dataset_dir}/train.txt", ent2idx, rel2idx)
    h, t = train_triples[:, 0], train_triples[:, 2]
    n = numerical_literals[h, :] - numerical_literals[t, :]
    c = np.mean(n, axis=0).astype("float32")
    var = np.var(n, axis=0).astype("float32") + 1e-6

    return (
        torch.autograd.Variable(torch.from_numpy(numerical_literals)),
        torch.autograd.Variable(torch.from_numpy(c)),
        torch.autograd.Variable(torch.from_numpy(var)),
    )


class _KBLNBase:
    kbln_model_name = None

    def _init_kbln(self, args, ent2idx=None, rel2idx=None):
        self.name = self.kbln_model_name or self.__class__.__name__
        self.literal_dir = args["dataset_dir"] + "/literals/"
        self.ent2idx = ent2idx
        self.rel2idx = rel2idx
        self.num_entities = args["num_entities"]
        self.numerical_literals, self.c, self.var = load_num_lit_kbln(
            ent2idx=self.ent2idx,
            rel2idx=self.rel2idx,
            dataset_dir=args["dataset_dir"],
        )
        self.n_num_lit = self.numerical_literals.size(1)
        self.nf_weights = torch.nn.Embedding(args["num_relations"], self.n_num_lit)

    def _kbln_score(self, x: torch.LongTensor):
        head_ids = x[:, 0]
        rel_ids = x[:, 1]
        numerical_literals = self.numerical_literals.to(x.device)
        c = self.c.to(x.device)
        var = self.var.to(x.device)
        n_h = numerical_literals[head_ids.view(-1)]
        n_t = numerical_literals
        n = n_h.unsqueeze(1).repeat(1, self.num_entities, 1) - n_t
        phi = torch.exp(-((n - c) ** 2) / var)
        w_nf = self.nf_weights(rel_ids.view(-1, 1))
        return torch.bmm(phi, w_nf.transpose(1, 2)).squeeze(-1)

    def forward_k_vs_all(self, x: torch.LongTensor):
        return super().forward_k_vs_all(x) + self._kbln_score(x)


class DistMult_KBLN(_KBLNBase, DistMult):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.kbln_model_name = "DistMult_KBLN"
        self._init_kbln(args, ent2idx=ent2idx, rel2idx=rel2idx)


class ComplEx_KBLN(_KBLNBase, ComplEx):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.kbln_model_name = "ComplEx_KBLN"
        self._init_kbln(args, ent2idx=ent2idx, rel2idx=rel2idx)


class Keci_KBLN(_KBLNBase, Keci):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.kbln_model_name = "Keci_KBLN"
        self._init_kbln(args, ent2idx=ent2idx, rel2idx=rel2idx)


class OMult_KBLN(_KBLNBase, OMult):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.kbln_model_name = "OMult_KBLN"
        self._init_kbln(args, ent2idx=ent2idx, rel2idx=rel2idx)


class QMult_KBLN(_KBLNBase, QMult):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.kbln_model_name = "QMult_KBLN"
        self._init_kbln(args, ent2idx=ent2idx, rel2idx=rel2idx)


class TransE_KBLN(_KBLNBase, TransE):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.kbln_model_name = "TransE_KBLN"
        self._init_kbln(args, ent2idx=ent2idx, rel2idx=rel2idx)


class DeCaL_KBLN(_KBLNBase, DeCaL):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.kbln_model_name = "DeCaL_KBLN"
        self._init_kbln(args, ent2idx=ent2idx, rel2idx=rel2idx)


class DualE_KBLN(_KBLNBase, DualE):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.kbln_model_name = "DualE_KBLN"
        self._init_kbln(args, ent2idx=ent2idx, rel2idx=rel2idx)
