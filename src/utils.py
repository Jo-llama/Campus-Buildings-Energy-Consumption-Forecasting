from pathlib import Path

def get_project_root():
    """Zwraca ścieżkę do katalogu głównego projektu."""
    return Path(__file__).resolve().parents[1]

def get_data_path(*path_parts):
    """Ścieżka do folderu data."""
    return get_project_root().joinpath("data", *path_parts)

def get_results_path(*path_parts):
    """Ścieżka do folderu results, tworzy folder jeśli nie istnieje."""
    path = get_project_root().joinpath("results", *path_parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_models_path(*path_parts):
    """Ścieżka do folderu models, tworzy folder jeśli nie istnieje."""
    path = get_project_root().joinpath("models", *path_parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_features_data_path():
    """Ścieżka do pliku features_data_encoded.csv bez twardego wpisywania."""
    return get_project_root() / "data" / "processed" / "features_data_encoded.csv"

# Listy

pochodne_targetu = ['consumption_lag_1h', 'consumption_lag_24h', 'consumption_lag_168h',
       'consumption_rolling_mean_3h', 'consumption_rolling_std_3h',
       'consumption_rolling_mean_6h', 'consumption_rolling_std_6h',
       'consumption_rolling_mean_24h', 'consumption_rolling_std_24h', 'consumption_per_sqm', 'consumption_per_person']

target_col = "consumption"

id_cols = ["campus_id",
           "meter_id",
           "campus_id_building"]

