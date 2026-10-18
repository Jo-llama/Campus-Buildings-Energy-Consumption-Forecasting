import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Project root directory


def merge_all_datasets(
        df_bc: pd.DataFrame,
        df_weather: pd.DataFrame,
        df_events: pd.DataFrame,
        df_cal: pd.DataFrame,
        df_building_meta: pd.DataFrame,
        df_campus_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges all datasets into a single DataFrame for modeling.
    """

    print("Merging datasets...")

    # 1. Convert to datetime
    df = df_bc.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_weather = df_weather.copy()
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    df_events = df_events.copy()
    df_events['date'] = pd.to_datetime(df_events['date'])
    df_cal = df_cal.copy()
    df_cal['date'] = pd.to_datetime(df_cal['date'])

    print(f"  Consumption: {df.shape}")

    # 2. Merge consumption + building_meta (by meter_id)
    df = df.merge(
        df_building_meta,
        left_on='meter_id',
        right_on='id',
        how='left',
        suffixes=('', '_building')
    )
    print(f"  + Building meta: {df.shape}")

    # 3. Merge + campus_meta (by campus_id)
    df = df.merge(
        df_campus_meta,
        left_on='campus_id',
        right_on='id',
        how='left',
        suffixes=('', '_campus')
    )
    print(f"  + Campus meta: {df.shape}")

    # 4. Merge + weather (by campus_id and timestamp)
    df = df.merge(
        df_weather,
        on=['campus_id', 'timestamp'],
        how='left'
    )
    print(f"  + Weather: {df.shape}")

    # 5. Merge + calendar (by date)
    df['date'] = df['timestamp'].dt.date
    df_cal['date'] = df_cal['date'].dt.date
    df = df.merge(
        df_cal,
        on='date',
        how='left'
    )
    print(f"  + Calendar: {df.shape}")

    # 6. Drop redundant id columns
    cols_to_drop = ['id', 'id_building', 'id_campus', 'date']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

    print(f"✓ Merge complete. Shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")

    return df


def train_test_split_temporal(
        df: pd.DataFrame,
        test_size: float = 0.2,
        date_col: str = 'timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split — train on older data, test on the most recent.
    """

    df = df.sort_values(date_col)
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\nTemporal split:")
    print(f"  Train: {len(train_df):,} rows ({train_df[date_col].min()} to {train_df[date_col].max()})")
    print(f"  Test:  {len(test_df):,} rows ({test_df[date_col].min()} to {test_df[date_col].max()})")

    return train_df, test_df


def save_processed_data(df: pd.DataFrame, path: str = "data/processed/merged_data.csv"):
    """Saves the processed dataset."""
    # Use an absolute path
    full_path = PROJECT_ROOT / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)
    print(f"✓ Data saved to: {full_path}")

def load_processed_data(path: str = "data/processed/merged_data.csv") -> pd.DataFrame:
    """Loads the processed dataset."""
    # Use an absolute path
    full_path = PROJECT_ROOT / path
    print(f"Loading data from: {full_path}")
    df = pd.read_csv(full_path, parse_dates=['timestamp'])
    print(f"✓ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df




# def create_preprocessor():
#     """Creates the preprocessor only."""
#     numerical_simple = ['gross_floor_area', 'apparent_temperature',
#                         'air_temperature', 'dew_point_temperature', 'relative_humidity']
#     numerical_high_missing = ['room_area', 'capacity']
#     numerical_weather = ['wind_speed']
#     categorical_ordinal = ['built_year']
#     categorical_nominal = ['category', 'name']
#     binary_flags = ['is_holiday', 'is_semester', 'is_exam']
#     id_cols = ['campus_id', 'meter_id', 'timestamp']
#
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('simple_num', Pipeline([
#                 ('imputer', SimpleImputer(strategy='median')),
#                 ('scaler', StandardScaler())
#             ]), numerical_simple),
#             ('high_missing', Pipeline([
#                 ('imputer', KNNImputer(n_neighbors=5)),
#                 ('scaler', StandardScaler())
#             ]), numerical_high_missing),
#             ('weather', Pipeline([
#                 ('imputer', SimpleImputer(strategy='median')),
#                 ('scaler', StandardScaler())
#             ]), numerical_weather),
#             ('built_year', Pipeline([
#                 ('imputer', SimpleImputer(strategy='most_frequent')),
#                 ('scaler', StandardScaler())
#             ]), categorical_ordinal),
#             ('cat_nominal', Pipeline([
#                 ('imputer', SimpleImputer(strategy='most_frequent')),
#                 ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
#             ]), categorical_nominal),
#             ('binary', 'passthrough', binary_flags),
#             ('drop_cols', 'drop', id_cols)
#         ],
#         remainder='drop'
#     )
#     return preprocessor


# def add_informative_missing_flags(df_raw):
#     """Creates missing-value indicator flags."""
#     flags = pd.DataFrame(index=df_raw.index)
#     for col in ['room_area', 'capacity', 'capacity_campus', 'built_year']:
#         if col in df_raw.columns:
#             flags[f'{col}_missing'] = df_raw[col].isnull().astype(int)
#     return flags


# def process_data_pipeline(df_raw, save_path='data/processed/model_ready.parquet'):
#     """Main preprocessing pipeline function."""
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#
#     preprocessor = create_preprocessor()
#
#     X = df_raw.drop(['consumption'], axis=1)
#     y = df_raw['consumption'].reset_index(drop=True)
#
#     print("Preprocessing...", end=" ")
#     X_processed = preprocessor.fit_transform(X)
#
#     missing_flags = add_informative_missing_flags(df_raw)
#
#     feature_names = list(preprocessor.get_feature_names_out()) + missing_flags.columns.tolist()
#     X_final = pd.DataFrame(
#         np.hstack([X_processed, missing_flags.values]),
#         columns=feature_names,
#         index=df_raw.index
#     )
#
#     final_df = pd.concat([X_final.reset_index(drop=True), y], axis=1)
#     final_df.to_parquet(save_path)
#
#     os.makedirs('models', exist_ok=True)
#     joblib.dump(preprocessor, 'models/preprocessor.pkl')
#
#     print(f"✓ Done: {len(final_df)} records -> {save_path}")
#     return final_df, preprocessor
