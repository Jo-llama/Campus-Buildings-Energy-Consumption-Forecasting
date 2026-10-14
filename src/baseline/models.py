"""Krok 7.5: prognozy naiwne i LightGBM (uczenie na train, early stopping na val)."""
from typing import Dict, List, Optional

import lightgbm as lgb
import pandas as pd
from pandas.api.types import CategoricalDtype

from src.baseline import config
from src.baseline.features import feature_columns
from src.baseline.leakage import assert_no_forbidden


def naive_columns(horizon_h: int) -> Dict[str, str]:
    """Mapa nazwa_modelu -> kolumna lagu; duplikaty kolumn (np. h=24) są pomijane."""
    out: Dict[str, str] = {}
    for name, tmpl in config.NAIVE_LAGS_TMPL.items():
        col = tmpl.format(h=horizon_h)
        if col not in out.values():
            out[name] = col
    return out


def naive_predictions(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Prognozy naiwne jako kolumny (indeks jak df); lag jest już przesunięty o >= horyzont."""
    return pd.DataFrame({name: df[col] for name, col in naive_columns(horizon_h).items()},
                        index=df.index)


def categorical_dtypes(train: pd.DataFrame) -> Dict[str, CategoricalDtype]:
    """Słownik kategorii ustalany wyłącznie z train (nieznane w val/test -> NaN)."""
    return {c: CategoricalDtype(sorted(pd.unique(train[c].dropna()))) for c in config.CATEGORICAL_FEATURES}


def model_matrix(df: pd.DataFrame, horizon_h: int,
                 cat_dtypes: Dict[str, CategoricalDtype]) -> pd.DataFrame:
    """Macierz cech modelu z kategoriami wyrównanymi do słownika z train."""
    feats = feature_columns(horizon_h)
    assert_no_forbidden(feats)
    X = df[feats].copy()
    for c, dtype in cat_dtypes.items():
        X[c] = X[c].astype(object).astype(dtype)
    return X


def train_lgbm(train: pd.DataFrame, val: pd.DataFrame, horizon_h: int,
               params: Optional[Dict] = None,
               num_boost_round: int = config.LGBM_NUM_BOOST_ROUND,
               early_stopping: int = config.LGBM_EARLY_STOPPING) -> lgb.Booster:
    """Uczy LightGBM na train; val służy tylko do early stopping. Test nie jest argumentem.

    Słownik kategorii zapisuje się w `booster.cat_dtypes` (do predykcji).
    """
    params = {**config.LGBM_PARAMS, **(params or {})}
    cat_dtypes = categorical_dtypes(train)
    X_tr, X_va = (model_matrix(d, horizon_h, cat_dtypes) for d in (train, val))
    d_tr = lgb.Dataset(X_tr, train[config.TARGET], categorical_feature=list(cat_dtypes),
                       free_raw_data=False)
    d_va = lgb.Dataset(X_va, val[config.TARGET], reference=d_tr)
    booster = lgb.train(
        params, d_tr, num_boost_round=num_boost_round, valid_sets=[d_va], valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping, first_metric_only=True, verbose=False),
                   lgb.log_evaluation(period=200)],
    )
    booster.cat_dtypes = cat_dtypes
    return booster


def predict_lgbm(booster: lgb.Booster, df: pd.DataFrame, horizon_h: int) -> pd.Series:
    """Predykcja dla najlepszej iteracji (z val), indeks jak df."""
    X = model_matrix(df, horizon_h, booster.cat_dtypes)
    return pd.Series(booster.predict(X, num_iteration=booster.best_iteration),
                     index=df.index, name="lgbm")


def feature_importance(booster: lgb.Booster) -> pd.DataFrame:
    """Ważność cech (gain i split), posortowana malejąco po gain."""
    return pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance("gain"),
        "split": booster.feature_importance("split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)


def predictions_frame(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Klucze + target + predykcje wszystkich modeli (łączenie po kluczu, nie po pozycji)."""
    out = pd.concat([df[config.KEYS + [config.TARGET]], preds], axis=1)
    if out.duplicated(config.KEYS).any():
        raise ValueError("Zduplikowane klucze (meter_id, timestamp) w predykcjach")
    return out


def all_predictions(booster: lgb.Booster, df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Prognozy naiwne + LightGBM dla jednego splitu, na tych samych wierszach."""
    preds = naive_predictions(df, horizon_h)
    preds["lgbm"] = predict_lgbm(booster, df, horizon_h)
    return predictions_frame(df, preds)


def model_names(horizon_h: int) -> List[str]:
    """Nazwy kolumn predykcji w kolejności raportu."""
    return list(naive_columns(horizon_h)) + ["lgbm"]
