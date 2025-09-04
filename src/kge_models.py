import torch
from dicee.models import BaseKGE

import torch.nn as nn
import torch.nn.functional as F
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear


class CLNN(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = "CLNN"

        self.embedding_dim = args.get('embedding_dim', 32)
        dropout = args.get('dropout', 0.15)
        self.scoring_technique = args.get('scoring_technique', 'KvsAll')
        self.g = [-1]
        self.n_blades = 2 ** len(self.g)
        if self.scoring_technique == "NegSample":
            self.in_channels = self.embedding_dim *3 // self.n_blades
            self.out_channels = 1
        elif self.scoring_technique == "KvsAll":
            self.in_channels = self.embedding_dim * 2 // self.n_blades
            self.out_channels = self.num_entities
        else:
            raise ValueError(f"Unknown scoring technique: {self.scoring_technique}")


        
        self.input_layer = CliffordLinear(g=self.g, in_channels=self.in_channels,
             out_channels=self.in_channels, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm([self.in_channels, self.n_blades])
        self.output_layer = CliffordLinear(g=self.g,in_channels=self.in_channels,
            out_channels=self.out_channels, bias=True)
        
        # mixture weights (how much each layer contributes to training loss)
        # Learnable mixture parameter
        self.alpha_raw = nn.Parameter(torch.tensor(0.5))  # start at 0.5

    def get_input(self, x):
        if self.scoring_technique == "NegSample":
            emb_head, emb_rel, emb_tail = self.get_triple_representation(x)
            cat_inp = torch.cat((emb_head, emb_rel, emb_tail), dim=1)
        else:
            emb_head, emb_rel = self.get_head_relation_representation(x)
            cat_inp = torch.cat((emb_head, emb_rel), dim=1)
        # reshape for convolutional/structured processing
        cat_inp = cat_inp.view(-1, self.in_channels, self.n_blades)
        return cat_inp

    
    def forward(self,x):
        #clnnn_input = self.get_input(x)
        # emb_head, emb_rel, emb_tail = self.get_triple_representation(x)

        # clnnn_input = emb_head * emb_rel * emb_tail
        # clnnn_input = clnnn_input.view(-1, self.in_channels, self.n_blades)
        clnnn_input = self.get_input(x)
        z = F.relu(self.input_layer(clnnn_input))
        out = self.output_layer(z)
        if self.training:
            return z[:,:,0], out[:,:,0]   # shape: (batch, features)
        else:
            return out[:,:,0]

        

    def goodness_in(self, x):
        # Squared activation energy (per sample)
        return torch.sum(x ** 2, dim=1)   # shape: (batch,)
    
    def goodness_out(self, score):
        """Goodness for last layer = directly the scalar output"""
        return score.squeeze(1)  # (batch,)


    def ff_update(self, x_pos, x_neg, optimizer, threshold_in=2.0, threshold_out = 1.0):
        """
        Forward-Forward style update with margin/threshold goodness.
        - threshold: target value separating good (pos) and bad (neg) inputs
        - reg_lambda: embedding norm regularization weight
        """
        optimizer.zero_grad()

        in_pos, out_pos = self.forward(x_pos)
        in_neg, out_neg = self.forward(x_neg)

        #Per-layer goodness
        g1_pos, g1_neg = self.goodness_in(in_pos), self.goodness_in(in_neg)
        g2_pos, g2_neg = self.goodness_out(out_pos), self.goodness_out(out_neg)


        # Layer 1 encourages longer hidden activations for pos, shorter for neg
        loss1_pos = F.softplus(threshold_in - g1_pos).mean()
        loss1_neg = F.softplus(g1_neg - threshold_in).mean()

        # Layer 2 encourages final scalar score to be higher for pos, lower for neg
        loss2_pos = F.softplus(threshold_out - g2_pos).mean()
        loss2_neg = F.softplus(g2_neg - threshold_out).mean()


        # Total objective
        # Mixture weighted loss
        alpha = torch.sigmoid(self.alpha_raw)
        loss = alpha * (loss1_pos + loss1_neg) + (1-alpha) * (loss2_pos + loss2_neg)
        loss.backward()
        optimizer.step()

        return {
        "loss": loss.item(),
        "g1_pos": g1_pos.mean().item(),
        "g1_neg": g1_neg.mean().item(),
        "g2_pos": g2_pos.mean().item(),
        "g2_neg": g2_neg.mean().item()
    }

class DistMult(BaseKGE):
    """
    Embedding Entities and Relations for Learning and Inference in Knowledge Bases
    https://arxiv.org/abs/1412.6575"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'DistMult'

    def k_vs_all_score(self, emb_h: torch.FloatTensor, emb_r: torch.FloatTensor, emb_E: torch.FloatTensor):
        """

        Parameters
        ----------
        emb_h
        emb_r
        emb_E

        Returns
        -------

        """
        return torch.mm(self.hidden_dropout(self.hidden_normalizer(emb_h * emb_r)), emb_E.transpose(1, 0))

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_head, emb_rel = self.get_head_relation_representation(x)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel, emb_E=self.entity_embeddings.weight)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        # (b,d),     (b,d)
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        # (b, d)
        hr = torch.einsum('bd, bd -> bd', emb_head_real, emb_rel_real)
        # (b, k, d)
        t = self.entity_embeddings(target_entity_idx)
        return torch.einsum('bd, bkd -> bk', hr, t)


    def score(self, h, r, t):
        return (self.hidden_dropout(self.hidden_normalizer(h * r)) * t).sum(dim=1)
    
    def forward_ff(self, x: torch.LongTensor):
        """
        Forward pass for Forward-Forward training.
        Returns a vector representation of the triple instead of a scalar score.
        - x: (batch, 3) indices of (head, relation, tail)
        """
        emb_head, emb_rel, emb_tail = self.get_triple_representation(x)  # (batch, d) each
        triple_repr = (emb_head * emb_rel) * emb_tail                        # element-wise product
        return triple_repr 

    def ff_goodness(self, x: torch.LongTensor):
        # x: output of forward_ff, shape (batch, d)
        return torch.mean(x ** 2, dim=0)  # per-sample goodness

    def ff_update(self, x_pos, x_neg, optimizer, threshold=2.0, reg_lambda=1e-4, normalize_emb=True):
        """
        Forward-Forward style update with margin/threshold goodness.
        - threshold: target value separating good (pos) and bad (neg) inputs
        - reg_lambda: embedding norm regularization weight
        """
        optimizer.zero_grad()
        
        # vector repr → per-sample goodness
        g_pos = self.ff_goodness(self.forward(x_pos))  # (batch,)
        g_neg = self.ff_goodness(self.forward(x_neg))  # (batch,)


        threshold = 0.5 * (g_pos.mean().item() + g_neg.mean().item())
        # Soft margins: no dead zones
        loss_pos = F.softplus(threshold - g_pos).mean()   # positives ≥ threshold
        loss_neg = F.softplus(g_neg - threshold).mean()   # negatives ≤ threshold
        ff_loss  = loss_pos + loss_neg
        
        reg_loss = reg_lambda * sum(p.pow(2).sum() for p in self.parameters())
        total = ff_loss + reg_loss
        total.backward()
        optimizer.step()

        if normalize_emb:
            self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, p=2, dim=1)
            self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, p=2, dim=1)
        return {
            "loss": total.item(),
            "pos_goodness": g_pos.mean().item(),
            "neg_goodness": g_neg.mean().item()
        }