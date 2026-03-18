import pytest

from runners.kge_runner import train_kge_model
from src.config import get_default_arguments


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_clnn_kge_kvsall_minimal_run():
    """
    Smoke test: run CLNN_KGE with KvsAll using minimal required args.
    This validates that the KvsAll attention scoring path executes end-to-end.
    """
    args = get_default_arguments([])

    # Minimal required configuration (following runner syntax)
    args.dataset_dir = "KGs/UMLS"
    args.model = "CLNN_KGE"
    args.scoring_technique = "KvsAll"
    args.embedding_dim = 32
    args.num_epochs = 1
    args.batch_size = 1024
    args.lr = 0.01
    args.test_runs = True
    args.save_experiment = False

    try:
        train_kge_model(args)
    except Exception as exc:
        pytest.fail(f"CLNN_KGE minimal KvsAll run failed: {exc}")
