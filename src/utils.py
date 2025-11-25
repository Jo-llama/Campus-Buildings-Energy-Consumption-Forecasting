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
    """Ścieżka do pliku features_data.csv bez twardego wpisywania."""
    return get_project_root() / "data" / "processed" / "features_data.csv"

