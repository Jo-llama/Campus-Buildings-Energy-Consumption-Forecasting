import os
from pathlib import Path
from typing import Dict

import pandas as pd
import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


def download_unicon(use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Pobiera dataset UNICON z Kaggle lub wczytuje z cache.

    Args:
        use_cache (bool): Jeśli True, używa danych z folderu raw.
                         Jeśli False, pobiera zawsze z Kaggle.

    Returns:
        Dict[str, pd.DataFrame]: Słownik z wczytanymi dataframe'ami
    """

    # Sprawdź czy dane już istnieją lokalnie
    csv_files = list(DATA_RAW.glob("**/weather_data.csv"))

    if use_cache and len(csv_files) > 0:
        print("✓ dane z cache (data/raw)")
        # Dane są już pobrane, wczytaj je bezpośrednio
        # Szukamy plików w zagnieżdżonych folderach
        data_path = csv_files[0].parent
    else:
        print("↓ Pobieranie danych z Kaggle...")
        os.environ["KAGGLEHUB_CACHE"] = str(DATA_RAW)
        path = kagglehub.dataset_download("cdaclab/unicon")
        data_path = Path(path)
        print(f"✓ Pobrano do: {data_path}")

    # Wczytanie wszystkich plików
    datasets = {
        'weather': pd.read_csv(data_path / "weather_data.csv"),
        'consumption': pd.read_csv(data_path / "building_consumption.csv"),
        'events': pd.read_csv(data_path / "events.csv"),
        'calendar': pd.read_csv(data_path / "calender.csv"),
        'building_meta': pd.read_csv(data_path / "building_meta.csv"),
        'campus_meta': pd.read_csv(data_path / "campus_meta.csv")
    }

    print(f"✓ Wczytano {len(datasets)} plików CSV")
    for name, df in datasets.items():
        print(f"  - {name}: {df.shape[0]:,} wierszy, {df.shape[1]} kolumn")

    return datasets


if __name__ == "__main__":
    # Test pobierania - używa cache jeśli istnieje
    data = download_unicon(use_cache=True)

    # Aby wymusić pobieranie z Kaggle:
    # data = download_unicon(use_cache=False)
