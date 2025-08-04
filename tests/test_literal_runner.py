import pytest
import os
import tempfile
import torch
import pandas as pd
import shutil
from unittest.mock import patch, MagicMock
from dicee.config import Namespace
from dicee.executer import Execute
from dicee.knowledge_graph_embeddings import KGE
from src.config import get_default_arguments
from runners.literal_runner import train_literals
from src.static_funcs import load_model_components


class TestLiteralRunner:
    """Regression tests for the literal runner train_literals method."""

    @pytest.fixture(scope="class")
    def pretrained_kge_model(self):
        """Setup a pre-trained KGE model for literal training."""
        
        # Set up the arguments for training a KGE model first
        args = get_default_arguments([])  # Pass empty list to avoid parsing pytest args
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/Family"
        args.backend = "pandas"
        args.num_epochs = 5  # Reduced for faster testing
        args.batch_size = 512
        args.lr = 0.1
        args.embedding_dim = 32
        args.trainer = 'torchCPUTrainer'  # Force CPU
        args.storage_path = "Experiments"

        # Train the KGE model
        result = Execute(args).start()
        
        return result['path_experiment_folder'], args.dataset_dir

    @pytest.fixture
    def literal_training_args(self, pretrained_kge_model):
        """Setup arguments for literal training."""
        pretrained_path, dataset_dir = pretrained_kge_model
        
        args = get_default_arguments([])
        
        # Required arguments for literal training
        args.pretrained_kge_path = pretrained_path
        args.dataset_dir = dataset_dir
        args.literal_model = 'mlp'
        args.lit_norm = 'z-norm'
        args.lit_epochs = 5  # Reduced for faster testing
        args.batch_size = 512
        args.lit_lr = 0.01
        args.random_seed = 42
        args.num_literal_runs = 1  # Single run for testing
        args.skip_eval_literals = False
        args.save_experiment = False
        args.freeze_entity_embeddings = True
        args.gate_residual = True
        args.dropout = 0.15
        args.multi_regression = False
        args.full_storage_path = None  # Will be set automatically
        
        return args

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_train_literals_mlp_basic(self, literal_training_args):
        """Test basic literal training with MLP model."""
        args = literal_training_args
        args.literal_model = 'mlp'
        
        # Test that train_literals runs without errors
        try:
            train_literals(args)
            assert True, "train_literals completed successfully"
        except Exception as e:
            pytest.fail(f"train_literals failed with error: {e}")

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_train_literals_clifford_basic(self, literal_training_args):
        """Test basic literal training with Clifford model."""
        args = literal_training_args
        args.literal_model = 'clifford'
        
        # Test that train_literals runs without errors
        try:
            train_literals(args)
            assert True, "train_literals completed successfully"
        except Exception as e:
            pytest.fail(f"train_literals failed with error: {e}")

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_train_literals_multiple_runs(self, literal_training_args):
        """Test literal training with multiple runs."""
        args = literal_training_args
        args.num_literal_runs = 2
        args.literal_model = 'mlp'
        
        try:
            train_literals(args)
            assert True, "train_literals with multiple runs completed successfully"
        except Exception as e:
            pytest.fail(f"train_literals with multiple runs failed with error: {e}")

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_train_literals_unfrozen_embeddings(self, literal_training_args):
        """Test literal training with unfrozen entity embeddings."""
        args = literal_training_args
        args.freeze_entity_embeddings = False
        args.literal_model = 'mlp'
        
        try:
            train_literals(args)
            assert True, "train_literals with unfrozen embeddings completed successfully"
        except Exception as e:
            pytest.fail(f"train_literals with unfrozen embeddings failed with error: {e}")

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_train_literals_skip_evaluation(self, literal_training_args):
        """Test literal training with evaluation skipped."""
        args = literal_training_args
        args.skip_eval_literals = True
        args.literal_model = 'mlp'
        
        try:
            train_literals(args)
            assert True, "train_literals with skipped evaluation completed successfully"
        except Exception as e:
            pytest.fail(f"train_literals with skipped evaluation failed with error: {e}")

    