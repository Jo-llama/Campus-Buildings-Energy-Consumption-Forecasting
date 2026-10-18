"""Configuration constants for the honest baseline (paths, horizon, split dates)."""
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"

MERGED_CSV: Path = DATA_PROCESSED / "merged_data.csv"
HOURLY_PARQUET: Path = DATA_PROCESSED / "hourly.parquet"
EVENTS_PARQUET: Path = DATA_PROCESSED / "events.parquet"

SEED: int = 42

# Readings are every 15 min; an hour needs all 4 readings, otherwise target = NaN
READINGS_PER_HOUR: int = 4
RAW_FREQ: str = "15min"

TARGET: str = "consumption"
KEYS: list = ["meter_id", "timestamp"]

# Static columns (constant within a meter) -> "first" aggregation
STATIC_COLS: list = [
    "campus_id", "campus_id_building", "built_year", "category",
    "gross_floor_area", "room_area", "capacity", "name", "capacity_campus",
]
# Weather -> hourly mean (wind_direction separately, circular mean)
WEATHER_COLS: list = [
    "apparent_temperature", "air_temperature", "dew_point_temperature",
    "relative_humidity", "wind_speed",
]
WIND_DIR_COL: str = "wind_direction"
# Calendar flags (per day) -> "first"
CALENDAR_COLS: list = ["is_holiday", "is_semester", "is_exam"]

# --- 7.3: features ---
# Forecast horizon in hours. 1 = forecast for the next hour.
# 24 = day-ahead variant: only lags >= 24 remain, and rolling windows end at t-24.
HORIZON_H: int = 1
LAGS_H: list = [1, 2, 3, 24, 48, 168]
ROLL_WINDOWS_H: list = [3, 6, 24, 168]
# Base temperature for heating/cooling degree-hours (°C)
DEGREE_BASE_C: float = 18.0
# Model weather (observed at hour t = assumes a perfect weather forecast)
WEATHER_FEATURES: list = ["air_temperature", "apparent_temperature", "relative_humidity", "wind_speed"]
CATEGORICAL_FEATURES: list = ["meter_id", "category"]
FEATURES_PARQUET_TMPL: str = "features_h{h}.parquet"

# --- 7.4: time split (single source of truth for dates) ---
# End dates are inclusive (whole day); a GAP_DAYS-day gap after each part is dropped.
TRAIN_END: str = "2021-04-30"
VAL_END: str = "2021-10-31"
GAP_DAYS: int = 7
SPLITS_DIR: Path = DATA_PROCESSED / "splits"
# A row enters a split only if the target and the lags needed by the models/naive are not NaN
# (all models are scored on the same rows). {h} = horizon.
REQUIRED_NONNULL_TMPL: list = ["consumption", "lag_{h}h", "lag_24h", "lag_168h"]

# --- 7.5: baseline models (naive + LightGBM) ---
MODELS_DIR: Path = PROJECT_ROOT / "models" / "baseline"
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "baseline"
# Naive forecasts = a lag column; {h} = horizon (for h=24 naive_last == naive_day -> skipped)
NAIVE_LAGS_TMPL: dict = {"naive_last": "lag_{h}h", "naive_day": "lag_24h", "naive_week": "lag_168h"}
# Fixed parameters (no tuning) - tuning only after the honest baseline
LGBM_PARAMS: dict = {
    "objective": "regression",
    "metric": ["l1", "l2"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}
LGBM_NUM_BOOST_ROUND: int = 5000
# Early stopping on val only (first metric = l1); test takes no part in training
LGBM_EARLY_STOPPING: int = 100
