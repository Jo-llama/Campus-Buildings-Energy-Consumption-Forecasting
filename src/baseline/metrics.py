"""Krok 7.5: metryki regresji - globalne i per licznik, liczone na identycznych wierszach."""
from typing import Dict, List

import numpy as np
import pandas as pd

from src.baseline import config


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE, WAPE [%], R2 i bias (średnia y_pred - y_true).

    WAPE = sum|e| / sum|y| zamiast MAPE - odporny na godziny z zużyciem bliskim 0.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("NaN w y_true lub y_pred - modele muszą być oceniane na tych samych wierszach")
    err = y_pred - y_true
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    denom = np.abs(y_true).sum()
    return {
        "n": int(len(y_true)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wape_pct": float(100 * np.abs(err).sum() / denom) if denom > 0 else float("nan"),
        "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(err.mean()),
    }


def metrics_table(df: pd.DataFrame, pred_cols: List[str]) -> pd.DataFrame:
    """Metryki globalne: wiersz = model (kolumna predykcji), kolumny = metryki."""
    y = df[config.TARGET].to_numpy()
    rows = {c: regression_metrics(y, df[c].to_numpy()) for c in pred_cols}
    return pd.DataFrame.from_dict(rows, orient="index")


def per_meter_metrics(df: pd.DataFrame, pred_cols: List[str]) -> pd.DataFrame:
    """Metryki per licznik i model (format długi: meter_id, model, n, mae, ...)."""
    rows = []
    for meter, g in df.groupby("meter_id", sort=True):
        y = g[config.TARGET].to_numpy()
        for c in pred_cols:
            rows.append({"meter_id": meter, "model": c, **regression_metrics(y, g[c].to_numpy())})
    return pd.DataFrame(rows)


def skill_vs(table: pd.DataFrame, reference: str, metric: str = "mae") -> pd.Series:
    """Skill score = 1 - metric(model) / metric(reference); > 0 oznacza lepiej niż reference."""
    return 1 - table[metric] / table.loc[reference, metric]
