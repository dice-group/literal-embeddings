import torch
import torch.nn.functional as F
from dicee.models import BaseKGE

from .layers import FFLayer, FFOutLayer


class CLNN(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = "CLNN"

        self.g = [-1]
        self.n_blades = 2 ** (len(self.g))
        self.embedding_dim = args.get('embedding_dim', 32)
        self.in_channels = self.embedding_dim * 3 // self.n_blades
        self.hid_channels = self.embedding_dim * 3 // self.n_blades
        self.out_channels = 1

        self.in_layer = FFLayer(g=self.g, in_channels = self.in_channels, out_channels= self.hid_channels, lr = self.learning_rate  )
        self.out_layer = FFOutLayer(g=self.g, in_channels = self.hid_channels,out_channels= self.out_channels, lr = self.learning_rate )
        
    def reshape(self, x):
        B, total = x.shape
        if total % self.n_blades != 0:
            raise ValueError(f"Total channels {total} not divisible by n_blades {self.n_blades}")
        channels_per_blade = total // self.n_blades
        # reshape and permute to (B, channels_per_blade, n_blades)
        return x.view(B, self.n_blades, channels_per_blade).permute(0, 2, 1)
    

    def get_input(self, x):
        emb_head, emb_rel, emb_tail = self.get_triple_representation(x)
        cat_inp = torch.cat((emb_head, emb_rel, emb_tail), dim=1)
        return self.reshape(cat_inp)
        # B, total = cat_inp.shape
        # if total % self.n_blades != 0:
        #     raise ValueError(f"Total channels {total} not divisible by n_blades {self.n_blades}")
        # channels_per_blade = total // self.n_blades
        # # reshape and permute to (B, channels_per_blade, n_blades)
        # return cat_inp.view(B, self.n_blades, channels_per_blade).permute(0, 2, 1)

    def score(self, h,r,t):
        inp = torch.cat((h,r,t), dim=1)
        reshaped_inp = self.reshape(inp)
        hid = self.in_layer.forward(reshaped_inp)
        out = self.out_layer.forward(hid)
        return out.squeeze(1)[:,0]

    def ff_update(self, x_pos, x_neg, optimier):
        optimier.zero_grad()
        x_pos_inp = self.get_input(x_pos)
        x_pos_neg = self.get_input(x_neg)
        hid_pos, hid_neg, hid_loss = self.in_layer.train_step(x_pos_inp, x_pos_neg)
        out_loss = self.out_layer.train_step(hid_pos.detach(), hid_neg.detach())
        total_loss = hid_loss + out_loss
        return { "loss": total_loss }
   

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
    
class FFInternalLayer(nn.Module):
    """
    Internal Forward-Forward layer with Hinton-style contrastive loss.
    """
    def __init__(self, dim, lr=1e-3):
        super().__init__()
        self.W_h = nn.Linear(dim*2, dim)
        self.W_t = nn.Linear(dim*2, dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, h, r, t):
        h_next = F.relu(self.W_h(torch.cat([h, r], dim=-1)))
        t_next = F.relu(self.W_t(torch.cat([t, r], dim=-1)))
        return h_next, t_next

    def goodness(self, h_next, t_next):
        return (h_next * t_next).sum(dim=-1)

    def compute_loss(self, h_pos, r_pos, t_pos, h_neg, r_neg, t_neg):
        h_pos_next, t_pos_next = self.forward(h_pos, r_pos, t_pos)
        h_neg_next, t_neg_next = self.forward(h_neg, r_neg, t_neg)
        good_pos = self.goodness(h_pos_next, t_pos_next)
        good_neg = self.goodness(h_neg_next, t_neg_next)
        return -torch.log(torch.sigmoid(good_pos - good_neg)).mean()

    def step(self, h_pos, r_pos, t_pos, h_neg, r_neg, t_neg):
        loss = self.compute_loss(h_pos, r_pos, t_pos, h_neg, r_neg, t_neg)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class FFOutputLayer(nn.Module):
    """
    Output Forward-Forward layer trained with BCE, returns continuous scores for ranking.
    """
    def __init__(self, dim, lr=1e-3):
        super().__init__()
        self.W_h = nn.Linear(dim*2, dim)
        self.W_t = nn.Linear(dim*2, dim)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, h, r, t):
        """
        Returns continuous logits for ranking
        """
        h_next = F.relu(self.W_h(torch.cat([h, r], dim=-1)))
        t_next = F.relu(self.W_t(torch.cat([t, r], dim=-1)))
        logits = (h_next * t_next).sum(dim=-1)  # continuous score
        return logits

    def compute_loss(self, h_pos, r_pos, t_pos, h_neg, r_neg, t_neg):
        logits_pos = self.forward(h_pos, r_pos, t_pos)
        logits_neg = self.forward(h_neg, r_neg, t_neg)

        logits = torch.cat([logits_pos, logits_neg], dim=0)
        labels = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)], dim=0)
        loss = self.bce_loss(logits, labels)
        return loss

    def step(self, h_pos, r_pos, t_pos, h_neg, r_neg, t_neg):
        loss = self.compute_loss(h_pos, r_pos, t_pos, h_neg, r_neg, t_neg)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def score(self, h, r, t):
        """
        Returns continuous score for ranking/evaluation
        """
        logits = self.forward(h, r, t)
        return logits  # use raw logits or torch.sigmoid(logits)


class FFKGE(BaseKGE):
    """
    Two-layer FF-KGE model with separate optimizers and continuous scoring.
    """
    def __init__(self, args, lr1=1e-3, lr2=1e-3):
        super().__init__(args)
        self.name = "FFKGE"
        self.layer1 = FFInternalLayer(self.embedding_dim, lr=lr1)
        self.layer2 = FFOutputLayer(self.embedding_dim, lr=lr2)

    def ff_update(self, x_pos, x_neg, optimizer=None):
        # Get embeddings
        h_pos, r_pos, t_pos = self.get_triple_representation(x_pos)
        h_neg, r_neg, t_neg = self.get_triple_representation(x_neg)

        # --- Layer1 update ---
        loss1 = self.layer1.step(
            h_pos.detach(), r_pos.detach(), t_pos.detach(),
            h_neg.detach(), r_neg.detach(), t_neg.detach()
        )

        # --- Layer2 update ---
        h1_pos_out, t1_pos_out = self.layer1.forward(h_pos.detach(), r_pos.detach(), t_pos.detach())
        h1_neg_out, t1_neg_out = self.layer1.forward(h_neg.detach(), r_neg.detach(), t_neg.detach())

        loss2 = self.layer2.step(
            h1_pos_out.detach(), r_pos.detach(), t1_pos_out.detach(),
            h1_neg_out.detach(), r_neg.detach(), t1_neg_out.detach()
        )

        total_loss = loss1 + loss2
        return {"loss": total_loss}

    def score(self,h,r,t):
        """
        Compute continuous score for a single triple.
        """
        # Forward through layer1
        h1, t1 = self.layer1.forward(h, r, t)
        # Forward through layer2 for score
        score = self.layer2.score(h1, r, t1)
        return score