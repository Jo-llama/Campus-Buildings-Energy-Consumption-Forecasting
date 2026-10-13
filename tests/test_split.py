"""Testy kroku 7.4: przerwy między splitami, brak nakładania, oczekiwane liczniki."""
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.baseline import config
from src.baseline.features import build_features
from src.baseline.split import make_splits, split_bounds

GAP = pd.Timedelta(days=config.GAP_DAYS)


def _check_gaps_and_keys(splits: Dict[str, pd.DataFrame]) -> None:
    """Wspólne asercje: przerwa > gap między splitami, unikalne klucze, sort po czasie."""
    tr, va, te = splits["train"], splits["val"], splits["test"]
    assert tr["timestamp"].max() + GAP < va["timestamp"].min()
    assert va["timestamp"].max() + GAP < te["timestamp"].min()
    for s in splits.values():
        assert {"meter_id", "timestamp"} <= set(s.columns)
        assert not s.duplicated(["meter_id", "timestamp"]).any()
        assert s["timestamp"].is_monotonic_increasing
        assert s[config.TARGET].notna().all()


def _features_toy() -> pd.DataFrame:
    """3 liczniki, godzinowo 2021-03-01..2022-01-31, z brakami w targecie."""
    rng = np.random.default_rng(42)
    ts = pd.date_range("2021-03-01", "2022-01-31 23:00", freq="h")
    frames = []
    for m in (1, 2, 3):
        cons = rng.gamma(2.0, 5.0, len(ts))
        cons[rng.random(len(ts)) < 0.05] = np.nan
        frames.append(pd.DataFrame({
            "meter_id": m, "timestamp": ts, "consumption": cons, "category": "other",
            "gross_floor_area": np.nan, "room_area": np.nan, "capacity": np.nan,
            "built_year": np.nan, "capacity_campus": 1, "air_temperature": 15.0,
            "apparent_temperature": 15.0, "relative_humidity": 50.0, "wind_speed": 1.0,
            "is_holiday": 0.0, "is_semester": 1.0, "is_exam": 0.0,
        }))
    return build_features(pd.concat(frames, ignore_index=True), horizon_h=1)


def test_split_bounds_match_spec():
    b = split_bounds("2021-04-30", "2021-10-31", 7)
    assert b["train"][1] == pd.Timestamp("2021-05-01")
    assert b["val"][0] == pd.Timestamp("2021-05-08")
    assert b["val"][1] == pd.Timestamp("2021-11-01")
    assert b["test"][0] == pd.Timestamp("2021-11-08")


def test_toy_splits_gaps_meters_and_dropped_rows():
    f = _features_toy()
    splits = make_splits(f, "2021-04-30", "2021-10-31", gap_days=7)
    _check_gaps_and_keys(splits)
    for s in splits.values():
        assert sorted(s["meter_id"].unique()) == [1, 2, 3]
        assert s[["lag_1h", "lag_24h", "lag_168h"]].notna().all().all()
    # Dni przerwy nie występują w żadnym splicie
    all_ts = pd.concat([s["timestamp"] for s in splits.values()])
    assert not all_ts.between("2021-05-01", "2021-05-07 23:00").any()
    assert not all_ts.between("2021-11-01", "2021-11-07 23:00").any()
    # Usunięte są tylko wiersze z NaN w targecie/wymaganych lagach
    for s in splits.values():
        r = s.attrs["report"]
        assert r["rows_window"] - r["rows_kept"] == r["rows_lost"] >= r["lost_target_nan"]


def test_real_splits_gaps_and_expected_meters(hourly_real):
    f = build_features(hourly_real, horizon_h=1)
    splits = make_splits(f, config.TRAIN_END, config.VAL_END, config.GAP_DAYS)
    _check_gaps_and_keys(splits)
    assert splits["train"]["meter_id"].nunique() == hourly_real["meter_id"].nunique()
    # Oczekiwane liczniki liczone niezależnie z hourly: >= 200 h z targetem w oknie -> musi być
    # w splicie; brak targetu w oknie -> nie może być
    for name, (start, stop) in split_bounds(config.TRAIN_END, config.VAL_END, config.GAP_DAYS).items():
        win = hourly_real[(hourly_real["timestamp"] >= start) & (hourly_real["timestamp"] < stop)]
        valid = win.dropna(subset=[config.TARGET]).groupby("meter_id").size()
        got = set(splits[name]["meter_id"].unique())
        assert set(valid[valid >= 200].index) <= got, name
        assert got <= set(valid.index), name


@pytest.mark.parametrize("gap_days", [1, 7, 14])
def test_gap_parameter_respected(gap_days):
    splits = make_splits(_features_toy(), "2021-04-30", "2021-10-31", gap_days=gap_days)
    g = pd.Timedelta(days=gap_days)
    assert splits["train"]["timestamp"].max() + g < splits["val"]["timestamp"].min()
    assert splits["val"]["timestamp"].max() + g < splits["test"]["timestamp"].min()
