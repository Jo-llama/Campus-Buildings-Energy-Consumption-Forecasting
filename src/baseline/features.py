"""Budowa cech uczciwego baseline'u (7.2: cechy statyczne, 7.3: cechy tylko z przeszłości)."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.baseline import config
from src.baseline.leakage import assert_no_forbidden

# Cechy statyczne budynku/kampusu - metadane, nie pochodne targetu
STATIC_NUMERIC: list = ["gross_floor_area", "room_area", "capacity", "built_year", "capacity_campus"]
# Flagi braków tylko tam, gdzie braki występują (capacity_campus jest zawsze znane)
MISSING_FLAG_COLS: list = ["gross_floor_area", "room_area", "capacity", "built_year"]
# Wartości-zaślepki w building_meta oznaczające "brak danych"
PLACEHOLDER_AS_MISSING: Dict[str, list] = {
    "gross_floor_area": [0],   # powierzchnia 0 m2 jest fizycznie niemożliwa
    "capacity": [0],
    "built_year": [1899],      # rok-zaślepka powtarzający się w wielu budynkach
}


def add_static_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Dodaje statyczne cechy budynku z flagami braków, bez usuwania wierszy.

    Zaślepki (0, 1899) -> NaN; dla kolumn z MISSING_FLAG_COLS flaga `<col>_missing` (int8).
    NaN zostaje w kolumnie (LightGBM obsługuje braki natywnie, bez imputacji przed splitem).
    Zwraca (df, raport: liczba liczników z zaślepką / z brakiem per kolumna).
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


# --- 7.3: cechy wyłącznie z przeszłości ---

def lags_for_horizon(horizon_h: int) -> List[int]:
    """Zwraca lagi dozwolone dla horyzontu: tylko >= horizon_h (plus sam horyzont)."""
    lags = sorted({lag for lag in config.LAGS_H if lag >= horizon_h} | {horizon_h})
    return lags


def _check_complete_grid(df: pd.DataFrame) -> None:
    """Sprawdza, że w obrębie licznika kolejne wiersze dzieli dokładnie 1 godzina."""
    step = df.groupby("meter_id", sort=False)["timestamp"].diff().dropna()
    if not (step == pd.Timedelta(hours=1)).all():
        raise ValueError("Dane nie leżą na kompletnej siatce godzinowej - shift(1) != 1h")


def _target_history_features(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Lagi i rolling mean/std liczone per licznik z serii przesuniętej o horizon_h."""
    g = df.groupby("meter_id", sort=False)[config.TARGET]
    out = {f"lag_{lag}h": g.shift(lag) for lag in lags_for_horizon(horizon_h)}
    # Okno kończy się w t - horizon_h (dla h=1: w t-1), nigdy nie obejmuje t
    past = g.shift(horizon_h)
    past_g = past.groupby(df["meter_id"], sort=False)
    for w in config.ROLL_WINDOWS_H:
        mp = max(2, w // 2)  # min. połowa okna obserwowana, inaczej NaN
        out[f"roll_mean_{w}h"] = past_g.transform(lambda s: s.rolling(w, min_periods=mp).mean())
        out[f"roll_std_{w}h"] = past_g.transform(lambda s: s.rolling(w, min_periods=mp).std())
    return pd.DataFrame(out, index=df.index)


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cechy kalendarzowe godziny t (znane z wyprzedzeniem) + kodowanie sin/cos."""
    ts = df["timestamp"]
    out = pd.DataFrame({
        "hour": ts.dt.hour.astype("int8"),
        "day_of_week": ts.dt.dayofweek.astype("int8"),
        "month": ts.dt.month.astype("int8"),
    }, index=df.index)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    # Kodowanie cykliczne jak w create_time_features
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
    """Pogoda w godzinie t + stopniogodziny grzania/chłodzenia (baza 18 °C)."""
    out = df[config.WEATHER_FEATURES].copy()
    t = df["air_temperature"]
    out["heating_degree_h"] = (config.DEGREE_BASE_C - t).clip(lower=0)
    out["cooling_degree_h"] = (t - config.DEGREE_BASE_C).clip(lower=0)
    return out


def build_features(df_hourly: pd.DataFrame, horizon_h: int = config.HORIZON_H) -> pd.DataFrame:
    """Buduje cechy tylko z przeszłości dla prognozy na horizon_h godzin do przodu.

    Wejście: hourly.parquet (kompletna siatka per licznik). Wyjście: klucze
    (meter_id, timestamp), target `consumption` i cechy z `feature_columns(horizon_h)`.
    Nie usuwa wierszy - NaN w lagach obsługuje split (7.4).
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
        raise ValueError(f"Brak cech w wyniku: {missing}")
    assert_no_forbidden(feats)
    return out


def feature_columns(horizon_h: int = config.HORIZON_H) -> List[str]:
    """Lista cech modelu (meter_id i category jako kategoryczne LightGBM)."""
    static = STATIC_NUMERIC + [f"{c}_missing" for c in MISSING_FLAG_COLS]
    calendar = ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
                "month_sin", "month_cos", "day_sin", "day_cos"] + config.CALENDAR_COLS
    weather = config.WEATHER_FEATURES + ["heating_degree_h", "cooling_degree_h"]
    lags = [f"lag_{lag}h" for lag in lags_for_horizon(horizon_h)]
    rolls = [f"roll_{s}_{w}h" for w in config.ROLL_WINDOWS_H for s in ("mean", "std")]
    return config.CATEGORICAL_FEATURES + static + calendar + weather + lags + rolls
