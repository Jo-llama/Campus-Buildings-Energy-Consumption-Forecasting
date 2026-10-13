"""Wczytywanie danych wejściowych (merged_data, events) i zapis Parquet."""
from pathlib import Path
from typing import Dict

import pandas as pd

from src.baseline import config


def _find_raw_dir() -> Path:
    """Zwraca katalog z surowymi CSV UNICON (szuka weather_data.csv w data/raw)."""
    hits = sorted(config.DATA_RAW.glob("**/weather_data.csv"))
    if hits:
        return hits[0].parent
    # Brak cache -> pobranie z Kaggle istniejącą funkcją (import leniwy, wymaga kagglehub)
    from data.download_kaggle import download_unicon
    download_unicon(use_cache=False)
    return sorted(config.DATA_RAW.glob("**/weather_data.csv"))[0].parent


def load_raw_tables() -> Dict[str, pd.DataFrame]:
    """Wczytuje surowe tabele UNICON potrzebne do merge_all_datasets."""
    raw = _find_raw_dir()
    files = {
        "consumption": "building_consumption.csv",
        "weather": "weather_data.csv",
        "events": "events.csv",
        "calendar": "calender.csv",
        "building_meta": "building_meta.csv",
        "campus_meta": "campus_meta.csv",
    }
    return {k: pd.read_csv(raw / f) for k, f in files.items()}


def load_merged(path: Path = config.MERGED_CSV) -> pd.DataFrame:
    """Wczytuje merged_data.csv; jeśli brak, odbudowuje go przez merge_all_datasets."""
    if not path.exists():
        from src.data_processing import merge_all_datasets
        t = load_raw_tables()
        df = merge_all_datasets(
            t["consumption"], t["weather"], t["events"], t["calendar"],
            t["building_meta"], t["campus_meta"],
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["category"] = df["category"].astype("category")
    df["name"] = df["name"].astype("category")
    return df


def load_events() -> pd.DataFrame:
    """Wczytuje surową tabelę events (meter_id, event_type, date, event_description)."""
    df = pd.read_csv(_find_raw_dir() / "events.csv", parse_dates=["date"])
    return df.sort_values(["meter_id", "date"]).reset_index(drop=True)


def load_calendar() -> pd.DataFrame:
    """Wczytuje surowy kalendarz (date, is_holiday, is_semester, is_exam) - pełne pokrycie dni."""
    return pd.read_csv(_find_raw_dir() / "calender.csv", parse_dates=["date"])


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Zapisuje DataFrame do Parquet (bez indeksu), tworząc katalog docelowy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
