from dicee.models.real import DistMult
from dicee.models.complex import ComplEx 
import torch
import torch.nn as nn
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
    df = pd.read_csv(f'{literal_dir}/numerical_literals.txt', header=None, sep='\t')
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

class DistMult_LiteralE(DistMult):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.name = 'DistMult_LiteralE'
        self.literal_dir = args["dataset_dir"] + '/literals/'
        self.ent2idx = ent2idx
        self.rel2idx = rel2idx

        self.numerical_literals = load_num_lit(self.ent2idx, self.literal_dir, wrap=True)
        self.n_num_lit = self.numerical_literals.size(1)
        self.emb_num_lit = Gate(self.embedding_dim + self.n_num_lit, self.embedding_dim)


    def forward_k_vs_all(self, x: torch.LongTensor):
        e_h = x[:, 0]
        numerical_literals = self.numerical_literals.to(x.device)
        emb_head, emb_rel = self.get_head_relation_representation(x)

        # Begin literals
        e1_lit = numerical_literals[e_h]
        emb_h = self.emb_num_lit(emb_head, e1_lit)
        emb_E = self.emb_num_lit(self.entity_embeddings.weight, numerical_literals)
        # End literals

        return super().k_vs_all_score(emb_h=emb_h, emb_r=emb_rel, emb_E=emb_E)
    

class ComplEx_LiteralE(ComplEx):
    def __init__(self, args, ent2idx=None, rel2idx=None):
        super().__init__(args)
        self.name = 'ComplEx_LiteralE'
        self.literal_dir = args["dataset_dir"] + '/literals/'
        self.ent2idx = ent2idx
        self.rel2idx = rel2idx


        self.numerical_literals = load_num_lit(self.ent2idx, self.literal_dir, wrap=True)
        self.n_num_lit = self.numerical_literals.size(1)
        self.emb_num_lit = Gate(self.embedding_dim + self.n_num_lit, self.embedding_dim)


    def forward_k_vs_all(self, x: torch.LongTensor):
        e_h = x[:, 0]
        numerical_literals = self.numerical_literals.to(x.device)
        emb_head, emb_rel = self.get_head_relation_representation(x)

        # Begin literals
        e1_lit = numerical_literals[e_h]
        emb_h = self.emb_num_lit(emb_head, e1_lit)
        emb_E = self.emb_num_lit(self.entity_embeddings.weight, numerical_literals)
        # End literals

        return super().k_vs_all_score(emb_h=emb_h, emb_r=emb_rel, emb_E=emb_E)
