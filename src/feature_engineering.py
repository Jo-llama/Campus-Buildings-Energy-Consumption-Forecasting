import pandas as pd
import numpy as np
from datetime import timedelta


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-based features from the timestamp column.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Basic time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_month'] = df['timestamp'].dt.day
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    df['quarter'] = df['timestamp'].dt.quarter

    # Cyclic features (sin/cos transformation to capture periodicity)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Binary flag features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)

    print(f"✓ Time features: {df.shape[1]} columns")

    return df


def create_lagged_features(df: pd.DataFrame, lags: list = [1, 24, 168]) -> pd.DataFrame:
    """
    Creates lagged features for consumption.

    Args:
        df: DataFrame
        lags: List of lag offsets (in hours)
    """
    df = df.copy()
    df = df.sort_values(['meter_id', 'timestamp'])

    for lag in lags:
        df[f'consumption_lag_{lag}h'] = df.groupby('meter_id')['consumption'].shift(lag)

    print(f"✓ Lag features: added {len(lags)} lags")

    return df


def create_rolling_features(df: pd.DataFrame, windows: list = [3, 6, 24]) -> pd.DataFrame:
    """
    Creates rolling statistics for consumption.

    Args:
        df: DataFrame
        windows: List of window sizes (in hours)
    """
    df = df.copy()
    df = df.sort_values(['meter_id', 'timestamp'])

    for window in windows:
        df[f'consumption_rolling_mean_{window}h'] = (
            df.groupby('meter_id')['consumption'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
        )
        df[f'consumption_rolling_std_{window}h'] = (
            df.groupby('meter_id')['consumption'].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
        )

    print(f"✓ Rolling features: added {len(windows) * 2} features")

    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived weather features.
    """
    df = df.copy()

    # Weather-derived features
    if 'apparent_temperature' in df.columns and 'air_temperature' in df.columns:
        df['temperature_diff'] = df['apparent_temperature'] - df['air_temperature']

    if 'relative_humidity' in df.columns:
        df['humidity_level'] = pd.cut(df['relative_humidity'],
                                      bins=[0, 30, 50, 70, 100],
                                      labels=['dry', 'moderate', 'humid', 'very_humid'],
                                      ordered=True).astype('category').cat.codes

    if 'wind_speed' in df.columns:
        df['wind_speed_squared'] = df['wind_speed'] ** 2

    print(f"✓ Weather features: derived features added")

    return df


def create_building_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features based on building information including construction decade dummies.
    """
    df = df.copy()

    if 'gross_floor_area' in df.columns and 'consumption' in df.columns:
        df['gross_floor_area'] = pd.to_numeric(df['gross_floor_area'], errors='coerce')
        df['consumption_per_sqm'] = df['consumption'] / (df['gross_floor_area'].replace(0, pd.NA) + 1e-6)

    if 'capacity' in df.columns and 'consumption' in df.columns:
        df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')
        df['consumption_per_person'] = df['consumption'] / (df['capacity'].replace(0, pd.NA) + 1e-6)

    if 'built_year' in df.columns:
        df['built_year_decade'] = df['built_year'].apply(
            lambda x: f"{int(x) // 10 * 10}s" if pd.notna(x) else np.nan
        )
        df = pd.get_dummies(df, columns=['built_year_decade'], prefix='decade')

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates interaction features between variables.
    """
    df = df.copy()

    # Temperature x humidity interaction
    if 'air_temperature' in df.columns and 'relative_humidity' in df.columns:
        df['temp_humidity_interaction'] = df['air_temperature'] * (df['relative_humidity'] / 100)

    # Day of week x hour interaction
    if 'day_of_week' in df.columns and 'hour' in df.columns:
        df['day_hour_interaction'] = df['day_of_week'] * df['hour']

    print(f"✓ Interaction features: interaction features added")

    return df


def apply_all_features(df: pd.DataFrame, lags: list = [1, 24, 168], windows: list = [3, 6, 24]) -> pd.DataFrame:
    """
    Applies all feature engineering transformations.
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    df = create_time_features(df)
    df = create_lagged_features(df, lags=lags)
    df = create_rolling_features(df, windows=windows)
    df = create_weather_features(df)
    df = create_building_features(df)
    df = create_interaction_features(df)

    # Drop NaN rows introduced by lag/rolling features
    df = df.dropna()

    print(f"\n✓ Feature engineering complete")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Missing values: {df.isnull().sum().sum()}")

    return df


def encode_categorical_features(df, categorical_cols):
    """
    Encodes categorical columns with one-hot encoding.

    Args:
        df (pd.DataFrame): input DataFrame
        categorical_cols (list): list of categorical column names

    Returns:
        pd.DataFrame: DataFrame with encoded features
    """
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded
