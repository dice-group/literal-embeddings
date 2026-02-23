import pytest
import torch

from runners.ff_runner import train_kge_ff
from src.config import get_default_arguments


class TestComplexFFPipeline:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_complex_ff_full_pipeline_10_epochs(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        args = get_default_arguments([])
        args.dataset_dir = "KGs/Family"
        args.model = "ComplexKGE"
        args.scoring_technique = "NegSample"
        args.num_epochs = 10
        args.batch_size = 256
        args.embedding_dim = 16
        args.lr = 0.05
        args.neg_ratio = 2
        args.ff_pos_weight = 1.0
        args.ff_neg_weight = 1.0
        args.ff_filtered_negatives = True
        args.ff_hard_negative_ratio = 0.0
        args.ff_max_filter_retries = 10
        args.eval_model = "val"
        args.test_runs = True
        args.save_every_n_epochs = False
        args.num_core = 0
        args.random_seed = 7

        report = train_kge_ff(args)

        assert isinstance(report, dict)
        assert "final_eval" in report
        assert isinstance(report["final_eval"], dict)
        assert len(report["final_eval"]) > 0

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_complex_ff_nobp_full_pipeline_smoke(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        args = get_default_arguments([])
        args.dataset_dir = "KGs/Family"
        args.model = "ComplexKGENoBP"
        args.scoring_technique = "NegSample"
        args.num_epochs = 3
        args.batch_size = 256
        args.embedding_dim = 16
        args.lr = 0.05
        args.neg_ratio = 2
        args.ff_pos_weight = 1.0
        args.ff_neg_weight = 1.0
        args.ff_filtered_negatives = True
        args.ff_hard_negative_ratio = 0.0
        args.ff_max_filter_retries = 10
        args.eval_model = "val"
        args.test_runs = True
        args.save_every_n_epochs = False
        args.num_core = 0
        args.random_seed = 11

        report = train_kge_ff(args)

        assert isinstance(report, dict)
        assert "final_eval" in report
        assert isinstance(report["final_eval"], dict)
        assert len(report["final_eval"]) > 0
