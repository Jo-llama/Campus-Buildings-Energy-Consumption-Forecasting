"""Krok 7.2: lista zakazanych (wyciekających) cech i strażnik sprawdzający kolumny."""
import re
from typing import Iterable, List

# Cechy liczone z targetu w starym pipeline (feature_engineering.py / 03-05) - nie odtwarzać:
# consumption / powierzchnia, consumption / osoby, komponenty PCA zbudowane m.in. na
# consumption_lag_* i rolling_* z bieżącym wierszem.
FORBIDDEN_PATTERNS: List[str] = [
    r"per_sqm",
    r"per_person",
    r"^pca_?\d*",
    r"^PC\d+$",
    r"^consumption$",           # sam target
    r"^consumption_",           # stare lag/rolling (złe jednostki, rolling z bieżącym wierszem)
    r"^n_valid_readings$",      # znana dopiero po zakończeniu godziny
]


def find_forbidden(columns: Iterable[str]) -> List[str]:
    """Zwraca kolumny pasujące do któregokolwiek wzorca z FORBIDDEN_PATTERNS."""
    rx = re.compile("|".join(f"(?:{p})" for p in FORBIDDEN_PATTERNS), flags=re.IGNORECASE)
    return [c for c in columns if rx.search(str(c))]


def assert_no_forbidden(columns: Iterable[str]) -> None:
    """Rzuca ValueError, jeśli lista cech zawiera zakazaną (wyciekającą) kolumnę."""
    bad = find_forbidden(columns)
    if bad:
        raise ValueError(f"Zakazane cechy (wyciek targetu): {bad}")
