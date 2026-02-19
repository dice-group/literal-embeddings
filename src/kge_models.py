import torch
import torch.nn as nn
import torch.nn.functional as F
from dicee.models import BaseKGE

from .layers import FFLayer, FFOutLayer


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
        x_pos_neg = self.get_input(x_neg)
        hid_pos, hid_neg, hid_loss, hid_loss_pos, hid_loss_neg = self.in_layer.train_step(x_pos_inp, x_pos_neg)

        # Step 1: update layer-1 with its local FF objective.
        optimizer.zero_grad()
        hid_loss.backward()
        optimizer.step()

        # Recompute hidden activations after layer-1 update for layer-2 training.
        with torch.no_grad():
            hid_pos = self.in_layer.forward(x_pos_inp)
            hid_neg = self.in_layer.forward(x_pos_neg)

        # Step 2: update layer-2 with fresh hidden states.
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

    def ff_update_kvsall(
        self,
        x_hr: torch.Tensor,
        y_vec: torch.Tensor,
        target_tails,
        optimizer,
        num_entities: int,
        num_negatives: int = 32,
        use_all_negatives: bool = False,
    ):
        """FF update from KvsAll-style (h,r)->tails batches.

        Converts multi-label tail targets into explicit positive/negative triples,
        then reuses the regular FF two-step layer-local update.
        """
        del y_vec  # target_tails already carries positive labels per (h,r)
        device = x_hr.device

        pos_chunks = []
        neg_chunks = []
        for i in range(x_hr.size(0)):
            h = int(x_hr[i, 0].item())
            r = int(x_hr[i, 1].item())
            pos_t = target_tails[i]
            if not isinstance(pos_t, torch.Tensor):
                pos_t = torch.as_tensor(pos_t, dtype=torch.long)
            pos_t = pos_t.long().to(device)
            if pos_t.numel() == 0:
                continue

            # Positive triples for all true tails of this (h,r).
            pos_hr = x_hr[i].unsqueeze(0).repeat(pos_t.numel(), 1)
            pos_chunk = torch.stack((pos_hr[:, 0], pos_hr[:, 1], pos_t), dim=1)
            pos_chunks.append(pos_chunk)

            # Build candidate negatives as non-target entities.
            is_target = torch.zeros(num_entities, dtype=torch.bool, device=device)
            is_target[pos_t] = True
            neg_candidates = torch.nonzero(~is_target, as_tuple=False).squeeze(1)
            if neg_candidates.numel() == 0:
                continue

            if use_all_negatives:
                sampled_neg_t = neg_candidates
            else:
                k = min(int(num_negatives) * max(1, pos_t.numel()), int(neg_candidates.numel()))
                perm = torch.randperm(neg_candidates.numel(), device=device)[:k]
                sampled_neg_t = neg_candidates[perm]

            neg_hr = x_hr[i].unsqueeze(0).repeat(sampled_neg_t.numel(), 1)
            neg_chunk = torch.stack((neg_hr[:, 0], neg_hr[:, 1], sampled_neg_t), dim=1)
            neg_chunks.append(neg_chunk)

        if len(pos_chunks) == 0 or len(neg_chunks) == 0:
            return {"loss": 0.0}

        x_pos = torch.cat(pos_chunks, dim=0)
        x_neg = torch.cat(neg_chunks, dim=0)
        return self.ff_update(x_pos, x_neg, optimizer)
   

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
