"""Shared fixtures: real data (skipped when hourly.parquet is missing)."""
import pandas as pd
import pytest

from src.baseline import config


@pytest.fixture(scope="session")
def hourly_real() -> pd.DataFrame:
    """Full hourly.parquet from step 7.1."""
    if not config.HOURLY_PARQUET.exists():
        pytest.skip("Missing data/processed/hourly.parquet - run --step resample")
    return pd.read_parquet(config.HOURLY_PARQUET)
