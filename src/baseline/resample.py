"""Krok 7.1: resampling 15-min -> 1h na kompletnej siatce godzinowej per licznik."""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.baseline import config


def clean_negative(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Ustawia ujemne zużycie (błąd licznika) na NaN; zwraca kopię i liczbę zmienionych wierszy."""
    df = df.copy()
    neg = df[config.TARGET] < 0
    df.loc[neg, config.TARGET] = np.nan
    return df, int(neg.sum())


def snap_to_grid(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Przesuwa odczyty spoza siatki 15-min (np. :29) do najbliższego slotu.

    Jeśli slot jest już zajęty przez odczyt z siatki, odczyt spoza siatki jest usuwany,
    więc w każdej godzinie są maksymalnie 4 odczyty.
    """
    df = df.copy()
    snapped = df["timestamp"].dt.round(config.RAW_FREQ)
    off_grid = df["timestamp"] != snapped
    df["timestamp"] = snapped
    df["_off_grid"] = off_grid
    # Odczyty z siatki mają pierwszeństwo przy kolizji
    df = df.sort_values(["meter_id", "timestamp", "_off_grid"], kind="mergesort")
    dup = df.duplicated(["meter_id", "timestamp"], keep="first")
    stats = {
        "off_grid": int(off_grid.sum()),
        "off_grid_dropped_collision": int(dup.sum()),
    }
    df = df.loc[~dup].drop(columns="_off_grid")
    return df, stats


def hour_label(ts: pd.Series) -> pd.Series:
    """Etykieta godziny dla odczytów oznaczonych końcem interwału.

    Odczyty 00:15, 00:30, 00:45, 01:00 opisują zużycie w godzinie 00:00-01:00,
    więc dostają etykietę 00:00 (początek godziny, closed='right', label='left').
    """
    return ts.dt.ceil("h") - pd.Timedelta(hours=1)


def _circular_mean_deg(sin_mean: pd.Series, cos_mean: pd.Series) -> pd.Series:
    """Średnia kołowa kierunku wiatru (w stopniach 0-360) z uśrednionych sin/cos."""
    deg = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
    # Wektor zerowy (brak danych) -> NaN
    return deg.where(sin_mean.notna() & cos_mean.notna())


def _complete_grid(hour_bounds: pd.DataFrame) -> pd.DataFrame:
    """Buduje pełną siatkę godzinową min->max dla każdego licznika."""
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
    """Agreguje dane 15-min do 1h na kompletnej siatce per licznik.

    - consumption: suma 4 odczytów, NaN gdy poprawnych odczytów < 4 (bez cichego 0),
    - pogoda: średnia godzinowa per kampus (wind_direction: średnia kołowa),
    - kolumny statyczne: first per licznik, flagi kalendarza: first per dzień
      (z tabeli `calendar` [date, is_*], jeśli podana - pokrywa też dni bez odczytów),
    - brakujące godziny zostają NaN w targecie (bez forward-fill).
    Zwraca (hourly, raport).
    """
    report: Dict = {"rows_in": int(len(df))}
    df, report["negative_set_nan"] = clean_negative(df)
    df, snap_stats = snap_to_grid(df)
    report.update(snap_stats)
    df["hour"] = hour_label(df["timestamp"])

    # Target: suma z min_count=4 -> NaN, gdy którykolwiek odczyt brakuje lub był ujemny
    g = df.groupby(["meter_id", "hour"], sort=True)[config.TARGET]
    cons = pd.DataFrame({
        config.TARGET: g.sum(min_count=config.READINGS_PER_HOUR),
        "n_valid_readings": g.count().astype("int8"),
    }).reset_index().rename(columns={"hour": "timestamp"})

    # Pełna siatka godzinowa per licznik (shift o 1 wiersz = 1 godzina)
    bounds = cons.groupby("meter_id")["timestamp"].agg(["min", "max"])
    grid = _complete_grid(bounds)
    hourly = grid.merge(cons, on=config.KEYS, how="left")
    hourly["n_valid_readings"] = hourly["n_valid_readings"].fillna(0).astype("int8")

    # Statyczne cechy budynku/kampusu (stałe w obrębie licznika)
    static = df.groupby("meter_id", sort=True)[config.STATIC_COLS].first().reset_index()
    hourly = hourly.merge(static, on="meter_id", how="left")

    # Pogoda per kampus i godzina (niezależna od licznika -> uzupełnia też godziny bez odczytów)
    w = df[["campus_id", "hour"] + config.WEATHER_COLS].copy()
    rad = np.radians(df[config.WIND_DIR_COL])
    w["_sin"], w["_cos"] = np.sin(rad), np.cos(rad)
    wh = w.groupby(["campus_id", "hour"], sort=True).mean().reset_index()
    wh[config.WIND_DIR_COL] = _circular_mean_deg(wh.pop("_sin"), wh.pop("_cos"))
    wh = wh.rename(columns={"hour": "timestamp"})
    hourly = hourly.merge(wh, on=["campus_id", "timestamp"], how="left")

    # Kalendarz per dzień (merge_all_datasets łączył po dacie)
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
    """Raport per licznik: liczba godzin, % NaN targetu, zakres dat."""
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
