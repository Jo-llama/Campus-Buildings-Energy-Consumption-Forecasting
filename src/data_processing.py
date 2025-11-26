import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Katalog główny projektu


def merge_all_datasets(
        df_bc: pd.DataFrame,
        df_weather: pd.DataFrame,
        df_events: pd.DataFrame,
        df_cal: pd.DataFrame,
        df_building_meta: pd.DataFrame,
        df_campus_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Łączy wszystkie datasety w jeden DataFrame do modelowania.
    """

    print("Łączenie datasetów...")

    # 1. Konwersja na datetime
    df = df_bc.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_weather = df_weather.copy()
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    df_events = df_events.copy()
    df_events['date'] = pd.to_datetime(df_events['date'])
    df_cal = df_cal.copy()
    df_cal['date'] = pd.to_datetime(df_cal['date'])

    print(f"  Consumption: {df.shape}")

    # 2. Merge consumption + building_meta (po meter_id)
    df = df.merge(
        df_building_meta,
        left_on='meter_id',
        right_on='id',
        how='left',
        suffixes=('', '_building')
    )
    print(f"  + Building meta: {df.shape}")

    # 3. Merge + campus_meta (po campus_id)
    df = df.merge(
        df_campus_meta,
        left_on='campus_id',
        right_on='id',
        how='left',
        suffixes=('', '_campus')
    )
    print(f"  + Campus meta: {df.shape}")

    # 4. Merge + weather (po campus_id i timestamp)
    df = df.merge(
        df_weather,
        on=['campus_id', 'timestamp'],
        how='left'
    )
    print(f"  + Weather: {df.shape}")

    # 5. Merge + calendar (po dacie)
    df['date'] = df['timestamp'].dt.date
    df_cal['date'] = df_cal['date'].dt.date
    df = df.merge(
        df_cal,
        on='date',
        how='left'
    )
    print(f"  + Calendar: {df.shape}")

    # 6. Cleanup kolumn
    cols_to_drop = ['id', 'id_building', 'id_campus', 'date']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

    print(f"✓ Merge zakończony. Wymiary: {df.shape}")
    print(f"  Kolumny ({len(df.columns)}): {list(df.columns)}")

    return df


def train_test_split_temporal(
        df: pd.DataFrame,
        test_size: float = 0.2,
        date_col: str = 'timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split - trening starszych danych, test na nowszych.
    """

    df = df.sort_values(date_col)
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\nTemporal split:")
    print(f"  Train: {len(train_df):,} wierszy ({train_df[date_col].min()} to {train_df[date_col].max()})")
    print(f"  Test:  {len(test_df):,} wierszy ({test_df[date_col].min()} to {test_df[date_col].max()})")

    return train_df, test_df


def save_processed_data(df: pd.DataFrame, path: str = "data/processed/merged_data.csv"):
    """Zapisuje przetworzony dataset."""
    # Użyj bezwzględnej ścieżki
    full_path = PROJECT_ROOT / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)
    print(f"✓ Dane zapisane do: {full_path}")

def load_processed_data(path: str = "data/processed/merged_data.csv") -> pd.DataFrame:
    """Wczytuje przetworzony dataset."""
    # Użyj bezwzględnej ścieżki
    full_path = PROJECT_ROOT / path
    print(f"Wczytywanie danych z: {full_path}")
    df = pd.read_csv(full_path, parse_dates=['timestamp'])
    print(f"✓ Wczytano {df.shape[0]:,} wierszy, {df.shape[1]} kolumn")
    return df
