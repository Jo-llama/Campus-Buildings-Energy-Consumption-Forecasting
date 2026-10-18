"""Step 7.6 tests: reproducing the old pipeline's leaks, the old split, the ablation."""
import numpy as np
import pandas as pd
import pytest

from src.baseline import config
from src.baseline.compare_old import (LEAK_RATIO, LEAK_ROLL, VARIANTS, ablation_table,
                                      add_old_leaky_features, old_positional_split, run_ablation)
from src.baseline.features import build_features, feature_columns
from src.baseline.leakage import find_forbidden
from src.baseline.models import train_lgbm

FAST = {"learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 20}


@pytest.fixture(scope="module")
def toy_features():
    """4 meters, daily profile + noise, a period covering the split dates from config."""
    rng = np.random.default_rng(0)
    ts = pd.date_range("2021-02-01", "2022-01-31 23:00", freq="h")
    profile = 10 * np.sin(2 * np.pi * ts.hour / 24) + 5 * (ts.dayofweek < 5)
    frames = []
    for m in (1, 2, 3, 4):
        cons = 10 * m + profile + rng.normal(0, 3, len(ts))
        frames.append(pd.DataFrame({
            "meter_id": m, "timestamp": ts, "consumption": cons, "category": "office",
            "gross_floor_area": 500.0 * m, "room_area": np.nan, "capacity": 20.0 * m,
            "built_year": 1990.0, "capacity_campus": 1, "air_temperature": 15.0,
            "apparent_temperature": 15.0, "relative_humidity": 50.0, "wind_speed": 1.0,
            "is_holiday": 0.0, "is_semester": 1.0, "is_exam": 0.0,
        }))
    return build_features(pd.concat(frames, ignore_index=True), horizon_h=1)


def test_leak_names_are_caught_by_guard():
    assert find_forbidden(LEAK_RATIO + LEAK_ROLL) == LEAK_RATIO + LEAK_ROLL


def test_ratio_and_rolling_leaks_contain_target(toy_features):
    f = add_old_leaky_features(toy_features)
    y = f[config.TARGET]
    # Ratio * building constant == target (exact leak)
    np.testing.assert_allclose(f["consumption_per_sqm"] * f["gross_floor_area"], y)
    # The window includes the current row: mean_3h(t) = (y_t + y_{t-1} + y_{t-2}) / 3
    m1 = f[f["meter_id"] == 1].reset_index(drop=True)
    t = 100
    expected = m1[config.TARGET].iloc[t - 2:t + 1].mean()
    assert m1["consumption_rolling_mean_3h"].iloc[t] == pytest.approx(expected)
    # The honest roll_mean_3h ends at t-1 and does not contain y_t
    assert m1["roll_mean_3h"].iloc[t] == pytest.approx(m1[config.TARGET].iloc[t - 3:t].mean())


def test_old_positional_split_is_a_meter_split(toy_features):
    s = old_positional_split(toy_features, 1)
    n = sum(len(p) for p in s.values())
    assert len(s["test"]) == n - int(n * 0.8)
    # Test = the meter(s) with the highest id, same period as train (no separation in time)
    assert s["test"]["meter_id"].min() >= s["train"]["meter_id"].max()
    assert s["test"]["timestamp"].min() < s["train"]["timestamp"].max()


def test_leaky_features_rejected_without_opt_in(toy_features):
    s = old_positional_split(add_old_leaky_features(toy_features), 1)
    with pytest.raises(ValueError, match="Forbidden"):
        train_lgbm(s["train"], s["val"], 1, params=FAST, num_boost_round=5,
                   features=feature_columns(1) + LEAK_RATIO)


def test_ablation_leaks_inflate_skill(toy_features):
    variants = {k: VARIANTS[k] for k in ("honest", "+ratio_leak", "+rolling_leak", "old_split")}
    res = run_ablation(toy_features, 1, variants, params=FAST, num_boost_round=300,
                       early_stopping=20)
    t = ablation_table(res)
    assert list(t.index) == list(variants)
    # Leaks give a skill close to 1 (the model reads y_t), the honest variant clearly less
    for leak in ("+ratio_leak", "+rolling_leak"):
        assert t.loc[leak, "skill_mae_vs_naive_last"] > t.loc["honest", "skill_mae_vs_naive_last"] + 0.2
    # Variants on the same split have identical rows and the same naive
    assert t.loc["honest", "n_test"] == t.loc["+ratio_leak", "n_test"]
    assert t.loc["honest", "naive_last_mae"] == t.loc["+ratio_leak", "naive_last_mae"]
