"""Loading input data (merged_data, events) and writing Parquet."""
from pathlib import Path
from typing import Dict

import pandas as pd

from src.baseline import config


def _find_raw_dir() -> Path:
    """Returns the directory with the raw UNICON CSVs (looks for weather_data.csv in data/raw)."""
    hits = sorted(config.DATA_RAW.glob("**/weather_data.csv"))
    if hits:
        return hits[0].parent
    # No cache -> download from Kaggle with the existing function (lazy import, needs kagglehub)
    from data.download_kaggle import download_unicon
    download_unicon(use_cache=False)
    return sorted(config.DATA_RAW.glob("**/weather_data.csv"))[0].parent


def load_raw_tables() -> Dict[str, pd.DataFrame]:
    """Loads the raw UNICON tables needed by merge_all_datasets."""
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
    """Loads merged_data.csv; if missing, rebuilds it with merge_all_datasets."""
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
    """Loads the raw events table (meter_id, event_type, date, event_description)."""
    df = pd.read_csv(_find_raw_dir() / "events.csv", parse_dates=["date"])
    return df.sort_values(["meter_id", "date"]).reset_index(drop=True)


def load_calendar() -> pd.DataFrame:
    """Loads the raw calendar (date, is_holiday, is_semester, is_exam) - covers every day."""
    return pd.read_csv(_find_raw_dir() / "calender.csv", parse_dates=["date"])


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Writes a DataFrame to Parquet (no index), creating the target directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
