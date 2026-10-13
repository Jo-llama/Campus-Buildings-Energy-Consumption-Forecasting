"""Wspólne fixtures: dane rzeczywiste (pomijane, gdy brak hourly.parquet)."""
import pandas as pd
import pytest

from src.baseline import config


@pytest.fixture(scope="session")
def hourly_real() -> pd.DataFrame:
    """Pełny hourly.parquet z kroku 7.1."""
    if not config.HOURLY_PARQUET.exists():
        pytest.skip("Brak data/processed/hourly.parquet - uruchom --step resample")
    return pd.read_parquet(config.HOURLY_PARQUET)
