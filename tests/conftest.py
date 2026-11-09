from __future__ import annotations

import pandas as pd
import pytest

from src.cli import main
from src.common.config import load_config

SAMPLE = "configs/sample.yaml"


@pytest.fixture(scope="session")
def cfg():
    return load_config(SAMPLE)


@pytest.fixture(scope="session")
def pipeline_done(cfg):
    """Exécute tout le pipeline une fois sur l'échantillon."""
    assert main(["all", "--config", SAMPLE]) == 0
    return cfg


@pytest.fixture()
def processed(pipeline_done) -> pd.DataFrame:
    return pd.read_parquet(pipeline_done.processed("dataset.parquet"))
