"""Feature building for the honest baseline (7.2: static features, 7.3: past-only features)."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.baseline import config
from src.baseline.leakage import assert_no_forbidden

# Static building/campus features - metadata, not derived from the target
STATIC_NUMERIC: list = ["gross_floor_area", "room_area", "capacity", "built_year", "capacity_campus"]
# Missing flags only where values can be missing (capacity_campus is always known)
MISSING_FLAG_COLS: list = ["gross_floor_area", "room_area", "capacity", "built_year"]
# Placeholder values in building_meta that mean "no data"
PLACEHOLDER_AS_MISSING: Dict[str, list] = {
    "gross_floor_area": [0],   # a floor area of 0 m2 is physically impossible
    "capacity": [0],
    "built_year": [1899],      # placeholder year repeated across many buildings
}


def add_static_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Adds static building features with missing flags, without dropping rows.

    Placeholders (0, 1899) -> NaN; columns in MISSING_FLAG_COLS get a `<col>_missing` flag (int8).
    NaN stays in the column (LightGBM handles missing values natively, no imputation before the split).
    Returns (df, report: number of meters with a placeholder / missing value per column).
    """
    df = df.copy()
    report: Dict[str, Dict[str, int]] = {}
    for col in STATIC_NUMERIC:
        placeholders = PLACEHOLDER_AS_MISSING.get(col, [])
        is_ph = df[col].isin(placeholders)
        meters = df.groupby("meter_id")
        report[col] = {
            "meters_placeholder": int(is_ph.groupby(df["meter_id"]).any().sum()),
            "meters_missing_before": int(meters[col].first().isna().sum()),
        }
        df[col] = df[col].mask(is_ph, np.nan).astype("float64")
        if col in MISSING_FLAG_COLS:
            df[f"{col}_missing"] = df[col].isna().astype("int8")
        report[col]["meters_missing_after"] = int(df.groupby("meter_id")[col].first().isna().sum())
    return df, report


# --- 7.3: past-only features ---

def lags_for_horizon(horizon_h: int) -> List[int]:
    """Returns the lags allowed for the horizon: only >= horizon_h (plus the horizon itself)."""
    lags = sorted({lag for lag in config.LAGS_H if lag >= horizon_h} | {horizon_h})
    return lags


def _check_complete_grid(df: pd.DataFrame) -> None:
    """Checks that consecutive rows within a meter are exactly 1 hour apart."""
    step = df.groupby("meter_id", sort=False)["timestamp"].diff().dropna()
    if not (step == pd.Timedelta(hours=1)).all():
        raise ValueError("Data is not on a complete hourly grid - shift(1) != 1h")


def _target_history_features(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Lags and rolling mean/std computed per meter from the series shifted by horizon_h."""
    g = df.groupby("meter_id", sort=False)[config.TARGET]
    out = {f"lag_{lag}h": g.shift(lag) for lag in lags_for_horizon(horizon_h)}
    # The window ends at t - horizon_h (for h=1: at t-1), never includes t
    past = g.shift(horizon_h)
    past_g = past.groupby(df["meter_id"], sort=False)
    for w in config.ROLL_WINDOWS_H:
        mp = max(2, w // 2)  # at least half the window observed, otherwise NaN
        out[f"roll_mean_{w}h"] = past_g.transform(lambda s: s.rolling(w, min_periods=mp).mean())
        out[f"roll_std_{w}h"] = past_g.transform(lambda s: s.rolling(w, min_periods=mp).std())
    return pd.DataFrame(out, index=df.index)


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features of hour t (known in advance) + sin/cos encoding."""
    ts = df["timestamp"]
    out = pd.DataFrame({
        "hour": ts.dt.hour.astype("int8"),
        "day_of_week": ts.dt.dayofweek.astype("int8"),
        "month": ts.dt.month.astype("int8"),
    }, index=df.index)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    # Cyclic encoding as in create_time_features
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["day_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["day_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    for c in config.CALENDAR_COLS:
        out[c] = df[c].astype("float32")
    return out


def _weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weather at hour t + heating/cooling degree-hours (base 18 °C)."""
    out = df[config.WEATHER_FEATURES].copy()
    t = df["air_temperature"]
    out["heating_degree_h"] = (config.DEGREE_BASE_C - t).clip(lower=0)
    out["cooling_degree_h"] = (t - config.DEGREE_BASE_C).clip(lower=0)
    return out


def build_features(df_hourly: pd.DataFrame, horizon_h: int = config.HORIZON_H) -> pd.DataFrame:
    """Builds past-only features for a forecast horizon_h hours ahead.

    Input: hourly.parquet (complete grid per meter). Output: keys
    (meter_id, timestamp), the `consumption` target and the features from `feature_columns(horizon_h)`.
    Does not drop rows - NaN in lags is handled by the split (7.4).
    """
    df = df_hourly.sort_values(config.KEYS, kind="mergesort").reset_index(drop=True)
    _check_complete_grid(df)
    static, _ = add_static_features(df[["meter_id"] + STATIC_NUMERIC])
    parts = [
        df[config.KEYS + [config.TARGET]],
        df[["category"]].astype("category"),
        static.drop(columns="meter_id"),
        _calendar_features(df),
        _weather_features(df),
        _target_history_features(df, horizon_h),
    ]
    out = pd.concat(parts, axis=1)
    out.attrs["horizon_h"] = horizon_h
    feats = feature_columns(horizon_h)
    missing = [c for c in feats if c not in out.columns]
    if missing:
        raise ValueError(f"Features missing from the output: {missing}")
    assert_no_forbidden(feats)
    return out


def feature_columns(horizon_h: int = config.HORIZON_H) -> List[str]:
    """Model feature list (meter_id and category as LightGBM categoricals)."""
    static = STATIC_NUMERIC + [f"{c}_missing" for c in MISSING_FLAG_COLS]
    calendar = ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
                "month_sin", "month_cos", "day_sin", "day_cos"] + config.CALENDAR_COLS
    weather = config.WEATHER_FEATURES + ["heating_degree_h", "cooling_degree_h"]
    lags = [f"lag_{lag}h" for lag in lags_for_horizon(horizon_h)]
    rolls = [f"roll_{s}_{w}h" for w in config.ROLL_WINDOWS_H for s in ("mean", "std")]
    return config.CATEGORICAL_FEATURES + static + calendar + weather + lags + rolls
