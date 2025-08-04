import pytest
import os
from torch.utils.data import DataLoader
from dicee.executer import Execute
from dicee.knowledge_graph_embeddings import KGE
from src.dataset import LiteralDataset
from src.model import LiteralEmbeddings, LiteralEmbeddingsClifford
from src.trainer_literal import train_literal_model
from src.config import get_default_arguments
from src.static_funcs import evaluate_lit_preds


class TestPredictLitRegression:
    """Regression tests for literal prediction using interactive KGE model Family dataset."""

    @pytest.fixture(scope="class")
    def family_model(self):
        """Setup Keci model trained on Family dataset."""
        
        # Set up the arguments for the Keci model
        args = get_default_arguments([])  # Pass empty list to avoid parsing pytest args
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/Family"
        args.backend = "pandas"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.lit_norm = "z-norm"
        args.lit_epochs = 10
        args.embedding_dim = 32
        args.trainer = 'torchCPUTrainer'  # Force CPU
        args.storage_path = "Experiments"  # Set default storage path

        result = Execute(args).start()
        kge_model = KGE(path=result['path_experiment_folder'])

        literal_dataset = LiteralDataset(
            dataset_dir=args.dataset_dir, 
            ent_idx=kge_model.entity_to_idx,
            normalization=args.lit_norm,
            sampling_ratio=0.5
        )
        literal_data_loader = DataLoader(
            literal_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(1, os.cpu_count())
        )
        # initialize literal model
        literal_model = LiteralEmbeddings(
            num_of_data_properties = literal_dataset.num_data_properties,
            embedding_dims = args.embedding_dim,
            entity_embeddings = kge_model.model.entity_embeddings,
            dropout = 0.15,
            freeze_entity_embeddings = True,
            gate_residual = True,
        )
        return kge_model.model, literal_model, literal_data_loader, literal_dataset, args

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_literal_training(self, family_model):
        """Test literal training with Keci model."""
        model, literal_model, literal_data_loader, literal_dataset, args = family_model
        

        # Train literal embeddings
        lit_model , lit_loss = train_literal_model(args=args, 
                                                   kge_model=model, 
                                                    literal_model=literal_model,
                                                   literal_batch_loader=literal_data_loader)
        assert lit_model is not None, "Literal model training failed"
        assert lit_loss is not None, "Literal loss should not be None"
        assert len(lit_loss["lit_loss"]) == args.lit_epochs, "Literal loss length should match number of epochs"

        lit_results = evaluate_lit_preds(
                literal_dataset,
                dataset_type="test",
                model=model,
                literal_model=literal_model,
                device=None,
                multi_regression=False
            )
        assert lit_results is not None, "Literal results should not be None"
        assert 'MAE' in lit_results.columns, "Literal results should contain 'MAE'"
        assert 'RMSE' in lit_results.columns, "Literal results should contain 'RMSE'"
        assert 'relation' in lit_results.columns, "Literal results should contain 'relation'"

        assert lit_results['MAE'].mean() > 0.1, "Mean Absolute Error should be a number"
        assert lit_results['RMSE'].mean() > 0.1, "Root Mean Squared Error should be a number"

class TestPredictLitCliffordRegression:
    """Regression tests for literal prediction using interactive KGE model Family dataset."""

    @pytest.fixture(scope="class")
    def family_model(self):
        """Setup Keci model trained on Family dataset."""
        
        # Set up the arguments for the Keci model
        args = get_default_arguments([])  # Pass empty list to avoid parsing pytest args
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/Family"
        args.backend = "pandas"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.lit_norm = "z-norm"
        args.lit_epochs = 10
        args.embedding_dim = 32
        args.trainer = 'torchCPUTrainer'  # Force CPU
        args.storage_path = "Experiments"  # Set default storage path

        result = Execute(args).start()
        kge_model = KGE(path=result['path_experiment_folder'])

        literal_dataset = LiteralDataset(
            dataset_dir=args.dataset_dir, 
            ent_idx=kge_model.entity_to_idx,
            normalization=args.lit_norm,
        )
        literal_data_loader = DataLoader(
            literal_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(1, os.cpu_count())
        )
        # initialize literal model
        literal_model = LiteralEmbeddingsClifford(
            num_of_data_properties = literal_dataset.num_data_properties,
            embedding_dims = args.embedding_dim,
            entity_embeddings = kge_model.model.entity_embeddings,
            dropout = 0.15,
            freeze_entity_embeddings = True,
            gate_residual = True,
        )
        return kge_model.model, literal_model, literal_data_loader, literal_dataset, args

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_literal_training(self, family_model):
        """Test literal training with Keci model."""
        model, literal_model, literal_data_loader, literal_dataset, args = family_model
        

        # Train literal embeddings
        lit_model , lit_loss = train_literal_model(args=args, 
                                                   kge_model=model, 
                                                    literal_model=literal_model,
                                                   literal_batch_loader=literal_data_loader)
        assert lit_model is not None, "Literal model training failed"
        assert lit_loss is not None, "Literal loss should not be None"
        assert len(lit_loss["lit_loss"]) == args.lit_epochs, "Literal loss length should match number of epochs"

        lit_results = evaluate_lit_preds(
                literal_dataset,
                dataset_type="test",
                model=model,
                literal_model=literal_model,
                device=None,
                multi_regression=False
            )
        assert lit_results is not None, "Literal results should not be None"
        assert 'MAE' in lit_results.columns, "Literal results should contain 'MAE'"
        assert 'RMSE' in lit_results.columns, "Literal results should contain 'RMSE'"
        assert 'relation' in lit_results.columns, "Literal results should contain 'relation'"

        assert lit_results['MAE'].mean() > 0.1, "Mean Absolute Error should be a number"
        assert lit_results['RMSE'].mean() > 0.1, "Root Mean Squared Error should be a number"