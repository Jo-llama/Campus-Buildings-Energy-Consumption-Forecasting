from pathlib import Path

def get_project_root():
    """Returns the path to the project root directory."""
    return Path(__file__).resolve().parents[1]

def get_data_path(*path_parts):
    """Returns path to the data folder."""
    return get_project_root().joinpath("data", *path_parts)

def get_results_path(*path_parts):
    """Returns path to the results folder, creating it if it does not exist."""
    path = get_project_root().joinpath("results", *path_parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_models_path(*path_parts):
    """Returns path to the models folder, creating it if it does not exist."""
    path = get_project_root().joinpath("models", *path_parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_features_data_path():
    """Returns path to features_data_encoded.csv without hardcoding."""
    return get_project_root() / "data" / "processed" / "features_data_encoded.csv"

# Lists

pochodne_targetu = ['consumption_lag_1h', 'consumption_lag_24h', 'consumption_lag_168h',
       'consumption_rolling_mean_3h', 'consumption_rolling_std_3h',
       'consumption_rolling_mean_6h', 'consumption_rolling_std_6h',
       'consumption_rolling_mean_24h', 'consumption_rolling_std_24h', 'consumption_per_sqm', 'consumption_per_person']

target_col = "consumption"

id_cols = ["campus_id",
           "meter_id",
           "campus_id_building"]
