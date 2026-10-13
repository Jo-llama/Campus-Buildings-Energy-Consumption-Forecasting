"""Budowa cech uczciwego baseline'u (7.2: cechy statyczne budynku)."""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Cechy statyczne budynku/kampusu - metadane, nie pochodne targetu
STATIC_NUMERIC: list = ["gross_floor_area", "room_area", "capacity", "built_year", "capacity_campus"]
# Wartości-zaślepki w building_meta oznaczające "brak danych"
PLACEHOLDER_AS_MISSING: Dict[str, list] = {
    "gross_floor_area": [0],   # powierzchnia 0 m2 jest fizycznie niemożliwa
    "capacity": [0],
    "built_year": [1899],      # rok-zaślepka powtarzający się w wielu budynkach
}


def add_static_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Dodaje statyczne cechy budynku z flagami braków, bez usuwania wierszy.

    Zaślepki (0, 1899) -> NaN; dla każdej kolumny flaga `<col>_missing` (int8).
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
        df[f"{col}_missing"] = df[col].isna().astype("int8")
        report[col]["meters_missing_after"] = int(df.groupby("meter_id")[col].first().isna().sum())
    return df, report
