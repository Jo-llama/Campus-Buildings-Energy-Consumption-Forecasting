"""Step 7.1: resampling 15-min -> 1h on a complete hourly grid per meter."""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.baseline import config


def clean_negative(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Sets negative consumption (meter error) to NaN; returns a copy and the number of changed rows."""
    df = df.copy()
    neg = df[config.TARGET] < 0
    df.loc[neg, config.TARGET] = np.nan
    return df, int(neg.sum())


def snap_to_grid(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Moves readings off the 15-min grid (e.g. :29) to the nearest slot.

    If the slot is already taken by an on-grid reading, the off-grid reading is dropped,
    so every hour has at most 4 readings.
    """
    df = df.copy()
    snapped = df["timestamp"].dt.round(config.RAW_FREQ)
    off_grid = df["timestamp"] != snapped
    df["timestamp"] = snapped
    df["_off_grid"] = off_grid
    # On-grid readings win on collision
    df = df.sort_values(["meter_id", "timestamp", "_off_grid"], kind="mergesort")
    dup = df.duplicated(["meter_id", "timestamp"], keep="first")
    stats = {
        "off_grid": int(off_grid.sum()),
        "off_grid_dropped_collision": int(dup.sum()),
    }
    df = df.loc[~dup].drop(columns="_off_grid")
    return df, stats


def hour_label(ts: pd.Series) -> pd.Series:
    """Hour label for readings stamped with the end of their interval.

    Readings 00:15, 00:30, 00:45, 01:00 describe consumption in the hour 00:00-01:00,
    so they get the label 00:00 (start of the hour, closed='right', label='left').
    """
    return ts.dt.ceil("h") - pd.Timedelta(hours=1)


def _circular_mean_deg(sin_mean: pd.Series, cos_mean: pd.Series) -> pd.Series:
    """Circular mean of wind direction (in degrees 0-360) from averaged sin/cos."""
    deg = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
    # Zero vector (no data) -> NaN
    return deg.where(sin_mean.notna() & cos_mean.notna())


def _complete_grid(hour_bounds: pd.DataFrame) -> pd.DataFrame:
    """Builds the full min->max hourly grid for each meter."""
    parts = [
        pd.DataFrame({
            "meter_id": meter_id,
            "timestamp": pd.date_range(row["min"], row["max"], freq="h"),
        })
        for meter_id, row in hour_bounds.iterrows()
    ]
    return pd.concat(parts, ignore_index=True)


def to_hourly(df: pd.DataFrame,
              calendar: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict]:
    """Aggregates 15-min data to 1h on a complete grid per meter.

    - consumption: sum of 4 readings, NaN when fewer than 4 valid readings (no silent 0),
    - weather: hourly mean per campus (wind_direction: circular mean),
    - static columns: first per meter, calendar flags: first per day
      (from the `calendar` table [date, is_*] if given - also covers days without readings),
    - missing hours stay NaN in the target (no forward-fill).
    Returns (hourly, report).
    """
    report: Dict = {"rows_in": int(len(df))}
    df, report["negative_set_nan"] = clean_negative(df)
    df, snap_stats = snap_to_grid(df)
    report.update(snap_stats)
    df["hour"] = hour_label(df["timestamp"])

    # Target: sum with min_count=4 -> NaN when any reading is missing or was negative
    g = df.groupby(["meter_id", "hour"], sort=True)[config.TARGET]
    cons = pd.DataFrame({
        config.TARGET: g.sum(min_count=config.READINGS_PER_HOUR),
        "n_valid_readings": g.count().astype("int8"),
    }).reset_index().rename(columns={"hour": "timestamp"})

    # Full hourly grid per meter (shift by 1 row = 1 hour)
    bounds = cons.groupby("meter_id")["timestamp"].agg(["min", "max"])
    grid = _complete_grid(bounds)
    hourly = grid.merge(cons, on=config.KEYS, how="left")
    hourly["n_valid_readings"] = hourly["n_valid_readings"].fillna(0).astype("int8")

    # Static building/campus features (constant within a meter)
    static = df.groupby("meter_id", sort=True)[config.STATIC_COLS].first().reset_index()
    hourly = hourly.merge(static, on="meter_id", how="left")

    # Weather per campus and hour (independent of meter -> also fills hours without readings)
    w = df[["campus_id", "hour"] + config.WEATHER_COLS].copy()
    rad = np.radians(df[config.WIND_DIR_COL])
    w["_sin"], w["_cos"] = np.sin(rad), np.cos(rad)
    wh = w.groupby(["campus_id", "hour"], sort=True).mean().reset_index()
    wh[config.WIND_DIR_COL] = _circular_mean_deg(wh.pop("_sin"), wh.pop("_cos"))
    wh = wh.rename(columns={"hour": "timestamp"})
    hourly = hourly.merge(wh, on=["campus_id", "timestamp"], how="left")

    # Calendar per day (merge_all_datasets joined on date)
    if calendar is not None:
        cal = (calendar.assign(_date=pd.to_datetime(calendar["date"]).dt.normalize())
                       .groupby("_date", sort=True)[config.CALENDAR_COLS].first())
    else:
        cal = (df.assign(_date=df["timestamp"].dt.normalize())
                 .groupby("_date", sort=True)[config.CALENDAR_COLS].first())
    hourly["_date"] = hourly["timestamp"].dt.normalize()
    hourly = hourly.join(cal, on="_date").drop(columns="_date")

    hourly = hourly.sort_values(config.KEYS, kind="mergesort").reset_index(drop=True)
    report["rows_out"] = int(len(hourly))
    report["hours_with_readings"] = int(len(cons))
    report["hours_incomplete_readings"] = int(((cons["n_valid_readings"] > 0)
                                               & (cons["n_valid_readings"] < 4)).sum())
    return hourly, report


def hourly_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per-meter report: number of hours, % NaN target, date range."""
    s = hourly.groupby("meter_id").agg(
        rows=(config.TARGET, "size"),
        nan_target_pct=(config.TARGET, lambda x: 100 * x.isna().mean()),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        category=("category", "first"),
        campus_id=("campus_id", "first"),
    )
    s["nan_target_pct"] = s["nan_target_pct"].round(2)
    return s
