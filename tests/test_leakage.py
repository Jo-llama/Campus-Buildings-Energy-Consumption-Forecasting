"""Testy gwarancji braku wycieku targetu (7.2; rozszerzane w 7.3)."""
import numpy as np
import pandas as pd
import pytest

from src.baseline.features import add_static_features
from src.baseline.leakage import assert_no_forbidden, find_forbidden

OLD_LEAKY = [
    "consumption_per_sqm", "consumption_per_person", "pca_1", "pca_6", "PC0",
    "consumption_rolling_mean_3h", "consumption_lag_1h", "consumption", "n_valid_readings",
]
HONEST = ["lag_1h", "roll_mean_24h", "air_temperature", "hour_sin", "capacity_missing", "meter_id"]


def test_guard_catches_old_leaky_features():
    assert find_forbidden(OLD_LEAKY) == OLD_LEAKY
    with pytest.raises(ValueError):
        assert_no_forbidden(HONEST + ["consumption_per_sqm"])


def test_guard_accepts_honest_features():
    assert_no_forbidden(HONEST)


def _static_toy() -> pd.DataFrame:
    """Dwa liczniki po 3 godziny: jeden z zaślepkami (0, 1899), drugi z brakami."""
    return pd.DataFrame({
        "meter_id": [1, 1, 1, 2, 2, 2],
        "consumption": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
        "gross_floor_area": [0.0] * 3 + [500.0] * 3,
        "room_area": [np.nan] * 3 + [100.0] * 3,
        "capacity": [0.0] * 3 + [np.nan] * 3,
        "built_year": [1899.0] * 3 + [1990.0] * 3,
        "capacity_campus": [10] * 6,
    })


def test_static_features_keep_rows_and_flag_missing():
    df, rep = add_static_features(_static_toy())
    assert len(df) == 6                                     # brak dropna
    assert df.loc[df.meter_id == 1, "gross_floor_area"].isna().all()
    assert df["gross_floor_area_missing"].tolist() == [1, 1, 1, 0, 0, 0]
    assert df["capacity_missing"].tolist() == [1] * 6
    assert df["built_year_missing"].tolist() == [1, 1, 1, 0, 0, 0]
    assert rep["capacity"]["meters_missing_after"] == 2


def test_static_features_do_not_depend_on_target():
    base, _ = add_static_features(_static_toy())
    toy = _static_toy()
    toy["consumption"] = toy["consumption"] * 100 + 7      # zmiana targetu
    changed, _ = add_static_features(toy)
    cols = [c for c in base.columns if c != "consumption"]
    pd.testing.assert_frame_equal(base[cols], changed[cols])
