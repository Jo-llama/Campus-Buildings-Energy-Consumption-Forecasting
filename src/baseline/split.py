"""Krok 7.4: split czasowy po dacie z przerwami (wszystkie liczniki w każdym splicie)."""
from typing import Dict, List, Tuple

import pandas as pd

from src.baseline import config

SPLIT_NAMES: Tuple[str, str, str] = ("train", "val", "test")


def split_bounds(train_end: str, val_end: str, gap_days: int = config.GAP_DAYS
                 ) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """Zwraca [start, koniec) dla train/val/test; daty końcowe włączne (cały dzień)."""
    gap = pd.Timedelta(days=gap_days)
    train_stop = pd.Timestamp(train_end) + pd.Timedelta(days=1)   # koniec dnia train_end
    val_start = train_stop + gap
    val_stop = pd.Timestamp(val_end) + pd.Timedelta(days=1)
    test_start = val_stop + gap
    return {
        "train": (pd.Timestamp.min, train_stop),
        "val": (val_start, val_stop),
        "test": (test_start, pd.Timestamp.max),
    }


def required_columns(horizon_h: int) -> List[str]:
    """Kolumny, które muszą być niepuste, żeby wiersz był użyty (target + lagi dla h)."""
    return list(dict.fromkeys(c.format(h=horizon_h) for c in config.REQUIRED_NONNULL_TMPL))


def make_splits(df_features: pd.DataFrame, train_end: str = config.TRAIN_END,
                val_end: str = config.VAL_END, gap_days: int = config.GAP_DAYS
                ) -> Dict[str, pd.DataFrame]:
    """Dzieli cechy po dacie na train/val/test z przerwami gap_days (przerwy usuwane).

    Sortuje globalnie po (timestamp, meter_id); usuwa tylko wiersze z NaN w targecie
    lub wymaganych lagach. Imputacja/skalowanie nie są potrzebne (LightGBM obsługuje NaN),
    więc nic nie jest dopasowywane na danych spoza train. meter_id i timestamp zostają
    kolumnami, żeby predykcje łączyć po kluczu, nigdy po pozycji.
    Raport (wiersze przed/po, utracone) jest w `splits[name].attrs["report"]`.
    """
    horizon_h = int(df_features.attrs.get("horizon_h", config.HORIZON_H))
    req = required_columns(horizon_h)
    df = df_features.sort_values(["timestamp", "meter_id"], kind="mergesort")
    splits: Dict[str, pd.DataFrame] = {}
    for name, (start, stop) in split_bounds(train_end, val_end, gap_days).items():
        part = df[(df["timestamp"] >= start) & (df["timestamp"] < stop)]
        kept = part.dropna(subset=req).reset_index(drop=True)
        kept.attrs = {"horizon_h": horizon_h, "report": {
            "rows_window": int(len(part)),
            "rows_kept": int(len(kept)),
            "rows_lost": int(len(part) - len(kept)),
            "lost_target_nan": int(part[config.TARGET].isna().sum()),
            "start": str(kept["timestamp"].min()),
            "end": str(kept["timestamp"].max()),
            "meters": int(kept["meter_id"].nunique()),
        }}
        splits[name] = kept
    return splits


def coverage_by_meter(splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Liczba wierszy per licznik w każdym splicie (0 = licznik nieobecny)."""
    cov = pd.concat({n: s.groupby("meter_id").size() for n, s in splits.items()}, axis=1)
    return cov.fillna(0).astype(int).sort_index()
