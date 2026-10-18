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
    Downloads the UNICON dataset from Kaggle or loads it from the cache.

    Args:
        use_cache (bool): If True, uses the data in the raw folder.
                         If False, always downloads from Kaggle.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of the loaded DataFrames
    """

    # Check whether the data already exists locally
    csv_files = list(DATA_RAW.glob("**/weather_data.csv"))

    if use_cache and len(csv_files) > 0:
        print("✓ data from cache (data/raw)")
        # Data is already downloaded, load it directly
        # Look for the files in nested folders
        data_path = csv_files[0].parent
    else:
        print("↓ Downloading data from Kaggle...")
        os.environ["KAGGLEHUB_CACHE"] = str(DATA_RAW)
        path = kagglehub.dataset_download("cdaclab/unicon")
        data_path = Path(path)
        print(f"✓ Downloaded to: {data_path}")

    # Load all files
    datasets = {
        'weather': pd.read_csv(data_path / "weather_data.csv"),
        'consumption': pd.read_csv(data_path / "building_consumption.csv"),
        'events': pd.read_csv(data_path / "events.csv"),
        'calendar': pd.read_csv(data_path / "calender.csv"),
        'building_meta': pd.read_csv(data_path / "building_meta.csv"),
        'campus_meta': pd.read_csv(data_path / "campus_meta.csv")
    }

    print(f"✓ Loaded {len(datasets)} CSV files")
    for name, df in datasets.items():
        print(f"  - {name}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    return datasets


if __name__ == "__main__":
    # Test the download - uses the cache if it exists
    data = download_unicon(use_cache=True)

    # To force a download from Kaggle:
    # data = download_unicon(use_cache=False)
