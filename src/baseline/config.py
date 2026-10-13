"""Stałe konfiguracyjne uczciwego baseline'u (ścieżki, horyzont, daty splitu)."""
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"

MERGED_CSV: Path = DATA_PROCESSED / "merged_data.csv"
HOURLY_PARQUET: Path = DATA_PROCESSED / "hourly.parquet"
EVENTS_PARQUET: Path = DATA_PROCESSED / "events.parquet"

SEED: int = 42

# Odczyty są co 15 min; godzina wymaga kompletu 4 odczytów, inaczej target = NaN
READINGS_PER_HOUR: int = 4
RAW_FREQ: str = "15min"

TARGET: str = "consumption"
KEYS: list = ["meter_id", "timestamp"]

# Kolumny statyczne (stałe w obrębie licznika) -> agregacja "first"
STATIC_COLS: list = [
    "campus_id", "campus_id_building", "built_year", "category",
    "gross_floor_area", "room_area", "capacity", "name", "capacity_campus",
]
# Pogoda -> średnia godzinowa (wind_direction osobno, średnia kołowa)
WEATHER_COLS: list = [
    "apparent_temperature", "air_temperature", "dew_point_temperature",
    "relative_humidity", "wind_speed",
]
WIND_DIR_COL: str = "wind_direction"
# Flagi kalendarzowe (per dzień) -> "first"
CALENDAR_COLS: list = ["is_holiday", "is_semester", "is_exam"]

# --- 7.3: cechy ---
# Horyzont prognozy w godzinach. 1 = prognoza na następną godzinę.
# 24 = wariant dobowy: zostają tylko lagi >= 24, a okna rolling kończą się w t-24.
HORIZON_H: int = 1
LAGS_H: list = [1, 2, 3, 24, 48, 168]
ROLL_WINDOWS_H: list = [3, 6, 24, 168]
# Bazowa temperatura stopniogodzin grzania/chłodzenia (°C)
DEGREE_BASE_C: float = 18.0
# Pogoda modelu (obserwowana w godzinie t = założenie idealnej prognozy pogody)
WEATHER_FEATURES: list = ["air_temperature", "apparent_temperature", "relative_humidity", "wind_speed"]
CATEGORICAL_FEATURES: list = ["meter_id", "category"]
FEATURES_PARQUET_TMPL: str = "features_h{h}.parquet"
