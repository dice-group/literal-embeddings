import torch
from dicee.models import BaseKGE

import torch.nn as nn
import torch.nn.functional as F
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear


class CLNN(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = "CLNN"

        n_layers = args.get('n_layer', 8)
        n_heads = args.get('n_head', 4)
        embedding_dim = args.get('embedding_dim', 32)
        in_features = embedding_dim * 2
        dropout = args.get('dropout', 0.15)
        inner_embedding_dim = args.get('inner_embedding_size', 4)
        self.scoring_technique = args.get('scoring_technique', 'KvsAll')
        self.g = [-1]
        self.n_blades = 2 ** len(self.g)
        if self.scoring_technique == "NegSample":
            self.in_channels = embedding_dim * 3 // self.n_blades
            self.out_channels = 1
        elif self.scoring_technique == "KvsAll":
            self.in_channels = embedding_dim * 2 // self.n_blades
            self.out_channels = self.num_entities
        else:
            raise ValueError(f"Unknown scoring technique: {self.scoring_technique}")

        assert n_layers > 0, "n_layer must be greater than 0"
        assert n_heads > 0, "n_head must be greater than 0"
        assert in_features % inner_embedding_dim == 0, (
            f"in_features ({in_features}) must be divisible by inner_embedding_size ({inner_embedding_dim})"
        )
        assert inner_embedding_dim % n_heads == 0, (
            f"inner_embedding_size ({inner_embedding_dim}) must be divisible by n_head ({n_heads})"
        )

        
        self.input_layer = CliffordLinear(
            g=self.g,
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            bias=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm([self.in_channels, self.n_blades])
        self.output_layer = CliffordLinear(
            g=self.g,
            in_channels=self.in_channels,
            out_channels=self.out_channels,  # for FF, keep output dim same as input
            bias=True,
        )
        self.threshold = 2.0

    def forward(self, x):
        if self.scoring_technique == "NegSample":
            emb_head, emb_rel, emb_tail = self.get_triple_representation(x)
            cat_inp = torch.cat((emb_head, emb_rel, emb_tail), dim=1)
        else:
            emb_head, emb_rel = self.get_head_relation_representation(x)
            cat_inp = torch.cat((emb_head, emb_rel), dim=1)

        # reshape for convolutional/structured processing
        clnnn_input = cat_inp.view(-1, self.in_channels, self.n_blades)
        
        z = F.relu(self.input_layer(clnnn_input))
        z = self.layer_norm(z)
        z = self.dropout(z)
        z = self.output_layer(z)

        # Instead of slicing only [:,:,0], pool across blades
        z = z[:,:,0]   # shape: (batch, features)
        return z

    def goodness(self, x):
        # Squared activation energy (per sample)
        return torch.mean(x ** 2, dim=1)   # shape: (batch,)

    def ff_update(self, x_pos, x_neg, optimizer, threshold=2.0, reg_lambda=1e-4):
        """
        Forward-Forward style update with margin/threshold goodness.
        - threshold: target value separating good (pos) and bad (neg) inputs
        - reg_lambda: embedding norm regularization weight
        """
        optimizer.zero_grad()

        g_pos = self.goodness(self.forward(x_pos))
        g_neg = self.goodness(self.forward(x_neg))

        # FF margin-style loss
        loss_pos = F.relu(threshold - g_pos).mean()   # positives should be >= threshold
        loss_neg = F.relu(g_neg - threshold).mean()   # negatives should be <= threshold
        loss = loss_pos + loss_neg

        # optional L2 regularization to prevent embedding explosion
        reg_loss = 0.0
        for p in self.parameters():
            reg_loss = reg_loss + p.norm(2).pow(2)
        reg_loss = reg_lambda * reg_loss

        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "pos_goodness": g_pos.mean().item(),
            "neg_goodness": g_neg.mean().item()
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