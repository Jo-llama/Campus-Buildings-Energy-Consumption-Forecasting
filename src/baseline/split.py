"""Step 7.4: date-based time split with gaps (all meters in every split)."""
from typing import Dict, List, Tuple

import pandas as pd

from src.baseline import config

SPLIT_NAMES: Tuple[str, str, str] = ("train", "val", "test")


def split_bounds(train_end: str, val_end: str, gap_days: int = config.GAP_DAYS
                 ) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """Returns [start, stop) for train/val/test; end dates are inclusive (whole day)."""
    gap = pd.Timedelta(days=gap_days)
    train_stop = pd.Timestamp(train_end) + pd.Timedelta(days=1)   # end of the train_end day
    val_start = train_stop + gap
    val_stop = pd.Timestamp(val_end) + pd.Timedelta(days=1)
    test_start = val_stop + gap
    return {
        "train": (pd.Timestamp.min, train_stop),
        "val": (val_start, val_stop),
        "test": (test_start, pd.Timestamp.max),
    }


def required_columns(horizon_h: int) -> List[str]:
    """Columns that must be non-null for a row to be used (target + lags for h)."""
    return list(dict.fromkeys(c.format(h=horizon_h) for c in config.REQUIRED_NONNULL_TMPL))


def make_splits(df_features: pd.DataFrame, train_end: str = config.TRAIN_END,
                val_end: str = config.VAL_END, gap_days: int = config.GAP_DAYS
                ) -> Dict[str, pd.DataFrame]:
    """Splits features by date into train/val/test with gap_days gaps (gaps are dropped).

    Sorts globally by (timestamp, meter_id); drops only rows with NaN in the target
    or required lags. No imputation/scaling is needed (LightGBM handles NaN),
    so nothing is fitted on data outside train. meter_id and timestamp stay
    columns so predictions are joined by key, never by position.
    The report (rows before/after, lost) is in `splits[name].attrs["report"]`.
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
    """Number of rows per meter in each split (0 = meter absent)."""
    cov = pd.concat({n: s.groupby("meter_id").size() for n, s in splits.items()}, axis=1)
    return cov.fillna(0).astype(int).sort_index()
