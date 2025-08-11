import torch
import numpy as np

class Evaluator:
    """Evaluates knowledge graph embedding models on link prediction tasks"""
    
    def __init__(self, er_vocab, train_datalaoder=None, test_dataloader=None, val_dataloader=None):
        self.er_vocab = er_vocab  # Dictionary mapping (head, relation) -> [valid_tails]
        self.train_dataloader = train_datalaoder
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader

    def evaluate(self, model, eval_mode='test', batch_size=1024, log=True):
        """Evaluate model on specified splits"""
        train_scores, test_scores, val_scores = None, None, None

        if 'train' in eval_mode:
            dataloader = self.train_dataloader
            train_scores = self.evaluate_link_prediction_performance(model, dataloader, batch_size=batch_size)

        if 'test' in eval_mode:
            dataloader = self.test_dataloader
            test_scores = self.evaluate_link_prediction_performance(model, dataloader, batch_size=batch_size)

        if 'val' in eval_mode:
            dataloader = self.val_dataloader
            val_scores = self.evaluate_link_prediction_performance(model, dataloader, batch_size=batch_size)

        if log:
            if train_scores is not None:
                print("Train Scores")
                print(train_scores)
            if test_scores is not None:
                print("Test Scores")
                print(test_scores)
            if val_scores is not None:
                print("Validation Scores")
                print(val_scores)
        
        eval_report = {}
        if train_scores is not None:
            eval_report['train'] = train_scores
        if test_scores is not None:
            eval_report['test'] = test_scores
        if val_scores is not None:
            eval_report['val'] = val_scores
        return eval_report
    
    @torch.no_grad()
    def evaluate_link_prediction_performance(self, model, data_loader, batch_size=1024):
        """Compute filtered link prediction metrics (MRR, Hits@1/3/10)"""
        model.eval()
        er_vocab = self.er_vocab
        hits_range = list(range(1, 11))
        hits = {i: [] for i in hits_range}
        ranks = []
        device = "cuda" if hasattr(model, "cuda") and model.cuda() and torch.cuda.is_available() else "cpu"
        num_triples = 0

        for batch in data_loader:
            e1_idx, r_idx, e2_idx = batch
            e1_idx, r_idx, e2_idx = (
                e1_idx.to(device), r_idx.to(device), e2_idx.to(device)
            )
            batch_size_actual = e1_idx.size(0)
            num_triples += batch_size_actual

            predictions = model(e1_idx, r_idx)  # [batch_size, num_entities]
            
            # Apply filtered evaluation
            for j in range(batch_size_actual):
                head_idx = e1_idx[j].item()
                rel_idx = r_idx[j].item()
                tail_idx = e2_idx[j].item()
                
                # Get valid tails for filtering
                filt = er_vocab.get((head_idx, rel_idx), [])
                filt = [idx for idx in filt if idx != tail_idx]
                
                # Store target score and filter out other valid tails
                target_value = predictions[j, tail_idx].item()
                if len(filt) > 0:
                    predictions[j, filt] = -np.Inf
                predictions[j, tail_idx] = target_value
                
            # Compute ranks
            _, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            for j in range(batch_size_actual):
                rank = torch.where(sort_idxs[j] == e2_idx[j])[0].item() + 1
                ranks.append(rank)
                for hits_level in hits_range:
                    hits[hits_level].append(1.0 if rank <= hits_level else 0.0)
                    
        # Compute final metrics
        hit_1 = np.mean(hits[1]) if num_triples > 0 else 0.0
        hit_3 = np.mean(hits[3]) if num_triples > 0 else 0.0
        hit_10 = np.mean(hits[10]) if num_triples > 0 else 0.0
        mean_reciprocal_rank = np.mean(1. / np.array(ranks)) if num_triples > 0 else 0.0

        results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10, 'MRR': mean_reciprocal_rank}
        return results