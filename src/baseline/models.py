"""Step 7.5: naive forecasts and LightGBM (trained on train, early stopping on val)."""
from typing import Dict, List, Optional

import lightgbm as lgb
import pandas as pd
from pandas.api.types import CategoricalDtype

from src.baseline import config
from src.baseline.features import feature_columns
from src.baseline.leakage import assert_no_forbidden


def naive_columns(horizon_h: int) -> Dict[str, str]:
    """Map model_name -> lag column; duplicate columns (e.g. h=24) are skipped."""
    out: Dict[str, str] = {}
    for name, tmpl in config.NAIVE_LAGS_TMPL.items():
        col = tmpl.format(h=horizon_h)
        if col not in out.values():
            out[name] = col
    return out


def naive_predictions(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Naive forecasts as columns (same index as df); the lag is already shifted by >= horizon."""
    return pd.DataFrame({name: df[col] for name, col in naive_columns(horizon_h).items()},
                        index=df.index)


def categorical_dtypes(train: pd.DataFrame) -> Dict[str, CategoricalDtype]:
    """Category vocabulary set from train only (unknown in val/test -> NaN)."""
    return {c: CategoricalDtype(sorted(pd.unique(train[c].dropna()))) for c in config.CATEGORICAL_FEATURES}


def model_matrix(df: pd.DataFrame, horizon_h: int, cat_dtypes: Dict[str, CategoricalDtype],
                 features: Optional[List[str]] = None, allow_leaky: bool = False) -> pd.DataFrame:
    """Model feature matrix with categories aligned to the train vocabulary.

    features=None -> feature_columns(horizon_h). allow_leaky=True disables the leakage guard -
    only for the old-pipeline ablation (7.6), never for a model reported as the baseline.
    """
    feats = list(features) if features is not None else feature_columns(horizon_h)
    if not allow_leaky:
        assert_no_forbidden(feats)
    X = df[feats].copy()
    for c, dtype in cat_dtypes.items():
        if c in X.columns:
            X[c] = X[c].astype(object).astype(dtype)
    return X


def train_lgbm(train: pd.DataFrame, val: pd.DataFrame, horizon_h: int,
               params: Optional[Dict] = None,
               num_boost_round: int = config.LGBM_NUM_BOOST_ROUND,
               early_stopping: int = config.LGBM_EARLY_STOPPING,
               features: Optional[List[str]] = None, allow_leaky: bool = False) -> lgb.Booster:
    """Trains LightGBM on train; val is used only for early stopping. Test is not an argument.

    The category vocabulary is stored in `booster.cat_dtypes` and the flag in `booster.allow_leaky`
    (for prediction; the feature list is in `booster.feature_name()`).
    """
    params = {**config.LGBM_PARAMS, **(params or {})}
    cat_dtypes = categorical_dtypes(train)
    X_tr, X_va = (model_matrix(d, horizon_h, cat_dtypes, features, allow_leaky) for d in (train, val))
    cats = [c for c in cat_dtypes if c in X_tr.columns]
    d_tr = lgb.Dataset(X_tr, train[config.TARGET], categorical_feature=cats,
                       free_raw_data=False)
    d_va = lgb.Dataset(X_va, val[config.TARGET], reference=d_tr)
    booster = lgb.train(
        params, d_tr, num_boost_round=num_boost_round, valid_sets=[d_va], valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping, first_metric_only=True, verbose=False),
                   lgb.log_evaluation(period=200)],
    )
    booster.cat_dtypes = cat_dtypes
    booster.allow_leaky = allow_leaky
    return booster


def predict_lgbm(booster: lgb.Booster, df: pd.DataFrame, horizon_h: int) -> pd.Series:
    """Prediction at the best iteration (from val), same index as df."""
    X = model_matrix(df, horizon_h, booster.cat_dtypes, booster.feature_name(),
                     getattr(booster, "allow_leaky", False))
    return pd.Series(booster.predict(X, num_iteration=booster.best_iteration),
                     index=df.index, name="lgbm")


def feature_importance(booster: lgb.Booster) -> pd.DataFrame:
    """Feature importance (gain and split), sorted by gain descending."""
    return pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance("gain"),
        "split": booster.feature_importance("split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)


def predictions_frame(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Keys + target + predictions of all models (joined by key, not by position)."""
    out = pd.concat([df[config.KEYS + [config.TARGET]], preds], axis=1)
    if out.duplicated(config.KEYS).any():
        raise ValueError("Duplicate keys (meter_id, timestamp) in predictions")
    return out


def all_predictions(booster: lgb.Booster, df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Naive + LightGBM forecasts for one split, on the same rows."""
    preds = naive_predictions(df, horizon_h)
    preds["lgbm"] = predict_lgbm(booster, df, horizon_h)
    return predictions_frame(df, preds)


def model_names(horizon_h: int) -> List[str]:
    """Prediction column names in report order."""
    return list(naive_columns(horizon_h)) + ["lgbm"]
