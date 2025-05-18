# tests/test_import_cf.py
from lightning_stpp.config_factory.benchmark_config import BenchmarkConfig

def test_benchmark_importable():
    assert BenchmarkConfig is not None
