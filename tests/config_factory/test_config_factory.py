# tests/test_config_factory.py
from pathlib import Path

from lightning_stpp.config_factory.benchmark_config import BenchmarkConfig



TEST_YAML = Path(__file__).with_suffix(".yaml")   # copy a tiny yaml next to this file

def test_yaml_loads_and_repr():
    cfg = BenchmarkConfig.from_yaml(TEST_YAML)
    # The __repr__ returns omegaconf yaml; make sure a key we expect is in there
    text = repr(cfg)
    assert "dataset:" in text
    assert cfg.dataset == "pinwheel"


def test_finalize_expands_paths(tmp_path):
    """tmp_path is a pytest built-in fixture giving an empty temp dir."""
    cfg = BenchmarkConfig.from_yaml(TEST_YAML, out_dir=str(tmp_path))
    cfg.finalize()

    # (1) out_dir auto-filled / created
    assert (tmp_path).exists()

    # (2) every experiment got the dataset name
    for exp in cfg.experiments:
        assert exp.data.name == cfg.dataset

    # (3) mlflow / ckpt prefixes were injected
        assert str(tmp_path) in exp.logging.mlflow_uri
        if exp.trainer.ckpt_dir:
            assert str(tmp_path) in exp.trainer.ckpt_dir


    
def test_repr_prints_yaml(capsys):
    cfg = BenchmarkConfig.from_yaml(TEST_YAML)
    print(cfg)                  # captured – not shown
    captured = capsys.readouterr()
    assert "experiments:" in captured.out

