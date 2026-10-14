"""Testy kroku 7.5: metryki, prognozy naiwne, LightGBM (tylko train/val, predykcje po kluczu)."""
import numpy as np
import pandas as pd
import pytest

from src.baseline import config
from src.baseline.features import build_features
from src.baseline.metrics import metrics_table, per_meter_metrics, regression_metrics, skill_vs
from src.baseline.models import (all_predictions, categorical_dtypes, model_matrix, model_names,
                                  naive_columns, train_lgbm)
from src.baseline.split import make_splits

FAST = {"learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 20}


@pytest.fixture(scope="module")
def toy_splits():
    """3 liczniki z profilem dobowym + szum; split jak w 7.4 (train/val/test)."""
    rng = np.random.default_rng(0)
    ts = pd.date_range("2021-01-01", "2022-01-31 23:00", freq="h")
    profile = 10 * np.sin(2 * np.pi * ts.hour / 24) + 5 * (ts.dayofweek < 5)
    frames = []
    for m, (base, cat) in enumerate([(20, "office"), (40, "lab"), (60, "office")], start=1):
        cons = base + profile + rng.normal(0, 2, len(ts))
        frames.append(pd.DataFrame({
            "meter_id": m, "timestamp": ts, "consumption": cons, "category": cat,
            "gross_floor_area": 1000.0 * m, "room_area": np.nan, "capacity": np.nan,
            "built_year": 1990.0, "capacity_campus": 1, "air_temperature": 15.0,
            "apparent_temperature": 15.0, "relative_humidity": 50.0, "wind_speed": 1.0,
            "is_holiday": 0.0, "is_semester": 1.0, "is_exam": 0.0,
        }))
    f = build_features(pd.concat(frames, ignore_index=True), horizon_h=1)
    return make_splits(f, "2021-09-30", "2021-11-30", gap_days=7)


@pytest.fixture(scope="module")
def toy_booster(toy_splits):
    return train_lgbm(toy_splits["train"], toy_splits["val"], 1, params=FAST,
                      num_boost_round=300, early_stopping=20)


def test_regression_metrics_hand_values():
    m = regression_metrics(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 2.0, 2.0, 4.0]))
    assert m["n"] == 4
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(np.sqrt(0.5))
    assert m["wape_pct"] == pytest.approx(100 * 2 / 10)
    assert m["r2"] == pytest.approx(1 - 2 / 5)
    assert m["bias"] == pytest.approx(0.0)


def test_metrics_reject_nan():
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0, np.nan]), np.array([1.0, 1.0]))


def test_skill_and_per_meter():
    df = pd.DataFrame({"meter_id": [1, 1, 2, 2], config.TARGET: [1.0, 3.0, 10.0, 20.0],
                       "a": [1.0, 3.0, 10.0, 20.0], "b": [2.0, 2.0, 12.0, 18.0]})
    t = metrics_table(df, ["a", "b"])
    assert skill_vs(t, "b")["a"] == pytest.approx(1.0)   # idealny model
    assert skill_vs(t, "b")["b"] == pytest.approx(0.0)
    pm = per_meter_metrics(df, ["a", "b"]).set_index(["meter_id", "model"])
    assert pm.loc[(1, "b"), "mae"] == pytest.approx(1.0)
    assert pm.loc[(2, "b"), "mae"] == pytest.approx(2.0)


def test_naive_columns_dedupe_for_daily_horizon():
    assert naive_columns(1) == {"naive_last": "lag_1h", "naive_day": "lag_24h", "naive_week": "lag_168h"}
    assert naive_columns(24) == {"naive_last": "lag_24h", "naive_week": "lag_168h"}


def test_categories_come_from_train_only(toy_splits):
    tr = toy_splits["train"]
    dtypes = categorical_dtypes(tr)
    other = toy_splits["test"].copy()
    other["meter_id"] = 99                       # licznik nieznany w train
    X = model_matrix(other, 1, dtypes)
    assert X["meter_id"].isna().all()
    assert list(X["meter_id"].cat.categories) == [1, 2, 3]
    assert list(X["category"].cat.categories) == ["lab", "office"]


def test_predictions_same_rows_keyed_and_finite(toy_splits, toy_booster):
    te = toy_splits["test"]
    pred = all_predictions(toy_booster, te, 1)
    assert list(pred.columns) == config.KEYS + [config.TARGET] + model_names(1)
    assert len(pred) == len(te)
    assert pred[model_names(1)].notna().all().all()
    # Naive = dokładnie kolumna lagu; target bez zmian
    assert (pred["naive_day"] == te["lag_24h"]).all()
    assert (pred[config.TARGET] == te[config.TARGET]).all()


def test_predictions_independent_of_row_order(toy_splits, toy_booster):
    te = toy_splits["test"]
    shuffled = te.sample(frac=1.0, random_state=1).reset_index(drop=True)
    a = all_predictions(toy_booster, te, 1).set_index(config.KEYS)["lgbm"]
    b = all_predictions(toy_booster, shuffled, 1).set_index(config.KEYS)["lgbm"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_early_stopping_uses_val_and_lgbm_beats_naive(toy_splits, toy_booster):
    assert 0 < toy_booster.best_iteration <= 300
    assert set(toy_booster.best_score) == {"val"}
    pred = all_predictions(toy_booster, toy_splits["test"], 1)
    t = metrics_table(pred, model_names(1))
    # Profil dobowy + szum iid: model uczy się średniej, naive powiela szum sprzed doby
    assert t.loc["lgbm", "mae"] < t.loc[["naive_last", "naive_day", "naive_week"], "mae"].min()

