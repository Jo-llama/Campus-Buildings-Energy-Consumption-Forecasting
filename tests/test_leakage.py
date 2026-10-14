"""Tests guaranteeing no target leakage (7.2; extended in 7.3)."""
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
    """Two meters with 3 hours each: one with placeholders (0, 1899), the other with missing values."""
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
    assert len(df) == 6                                     # no dropna
    assert df.loc[df.meter_id == 1, "gross_floor_area"].isna().all()
    assert df["gross_floor_area_missing"].tolist() == [1, 1, 1, 0, 0, 0]
    assert df["capacity_missing"].tolist() == [1] * 6
    assert df["built_year_missing"].tolist() == [1, 1, 1, 0, 0, 0]
    assert rep["capacity"]["meters_missing_after"] == 2


def test_static_features_do_not_depend_on_target():
    base, _ = add_static_features(_static_toy())
    toy = _static_toy()
    toy["consumption"] = toy["consumption"] * 100 + 7      # change the target
    changed, _ = add_static_features(toy)
    cols = [c for c in base.columns if c != "consumption"]
    pd.testing.assert_frame_equal(base[cols], changed[cols])


# --- 7.3: past-only features ---

from src.baseline.features import build_features, feature_columns  # noqa: E402


def _hourly_toy(n_hours: int = 400, seed: int = 42) -> pd.DataFrame:
    """Two meters on a complete hourly grid, with random gaps in the target."""
    rng = np.random.default_rng(seed)
    frames = []
    for m, cat in [(1, "teaching"), (2, "office")]:
        cons = rng.gamma(2.0, 5.0, n_hours)
        cons[rng.random(n_hours) < 0.05] = np.nan
        frames.append(pd.DataFrame({
            "meter_id": m,
            "timestamp": pd.date_range("2021-01-01", periods=n_hours, freq="h"),
            "consumption": cons,
            "category": cat, "gross_floor_area": 100.0 * m, "room_area": np.nan,
            "capacity": 0.0, "built_year": 1990.0, "capacity_campus": 1000,
            "air_temperature": rng.normal(15, 5, n_hours), "apparent_temperature": 14.0,
            "relative_humidity": 50.0, "wind_speed": 3.0,
            "is_holiday": 0.0, "is_semester": 1.0, "is_exam": 0.0,
        }))
    return pd.concat(frames, ignore_index=True)


def _assert_past_unchanged(hourly: pd.DataFrame, meter: int, t: pd.Timestamp,
                           new_value: float, horizon_h: int) -> None:
    """Changes the target at (meter, t) and checks that features at hours < t + horizon_h did not change."""
    feats = feature_columns(horizon_h)
    base = build_features(hourly, horizon_h)
    pert = hourly.copy()
    hit = (pert["meter_id"] == meter) & (pert["timestamp"] == t)
    assert hit.sum() == 1
    pert.loc[hit, "consumption"] = new_value
    changed = build_features(pert, horizon_h)

    # Same meter: all features up to and including t + horizon_h - 1 unchanged (for h=1: <= t)
    safe = (base["meter_id"] != meter) | (base["timestamp"] < t + pd.Timedelta(hours=horizon_h))
    pd.testing.assert_frame_equal(base.loc[safe, feats], changed.loc[safe, feats])
    # Check the test is not vacuous: the lag = horizon at t + horizon must change
    nxt = (base["meter_id"] == meter) & (base["timestamp"] == t + pd.Timedelta(hours=horizon_h))
    if nxt.any():
        a = base.loc[nxt, f"lag_{horizon_h}h"].iloc[0]
        b = changed.loc[nxt, f"lag_{horizon_h}h"].iloc[0]
        assert not (a == b or (np.isnan(a) and np.isnan(b)))


@pytest.mark.parametrize("horizon_h", [1, 24])
@pytest.mark.parametrize("hour_idx", [0, 5, 170, 399])
@pytest.mark.parametrize("new_value", [1e6, np.nan])
def test_perturbing_target_does_not_change_past_features_toy(horizon_h, hour_idx, new_value):
    hourly = _hourly_toy()
    hourly.loc[(hourly.meter_id == 1), "consumption"] = (
        hourly.loc[(hourly.meter_id == 1), "consumption"].fillna(1.0))  # t has a value to change
    t = pd.Timestamp("2021-01-01") + pd.Timedelta(hours=hour_idx)
    _assert_past_unchanged(hourly, meter=1, t=t, new_value=new_value, horizon_h=horizon_h)


def test_perturbing_target_does_not_change_past_features_real(hourly_real):
    rng = np.random.default_rng(42)
    meters = sorted(rng.choice(hourly_real["meter_id"].unique(), size=3, replace=False))
    sub = hourly_real[hourly_real["meter_id"].isin(meters)]
    rows = sub.dropna(subset=["consumption"]).sample(5, random_state=42)
    for _, r in rows.iterrows():
        _assert_past_unchanged(sub, int(r["meter_id"]), r["timestamp"], r["consumption"] * 3 + 10, 1)


def test_no_feature_near_perfect_spearman_with_target(hourly_real):
    f = build_features(hourly_real, 1)
    s = f.dropna(subset=["consumption"]).sample(300_000, random_state=42)
    num = [c for c in feature_columns(1) if c != "category"]
    rho = s[num].corrwith(s["consumption"], method="spearman").abs()
    assert rho.notna().all(), f"Constant features: {rho[rho.isna()].index.tolist()}"
    assert (rho <= 0.99).all(), rho[rho > 0.99].to_dict()


def test_horizon_24_uses_only_lags_ge_24():
    lags = [c for c in feature_columns(24) if c.startswith("lag_")]
    assert lags == ["lag_24h", "lag_48h", "lag_168h"]
    assert_no_forbidden(feature_columns(1) + feature_columns(24))
