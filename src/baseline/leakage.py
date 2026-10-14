"""Step 7.2: list of forbidden (leaking) features and a guard that checks columns."""
import re
from typing import Iterable, List

# Features computed from the target in the old pipeline (feature_engineering.py / 03-05) - do not recreate:
# consumption / floor area, consumption / people, PCA components built from, among others,
# consumption_lag_* and rolling_* that include the current row.
FORBIDDEN_PATTERNS: List[str] = [
    r"per_sqm",
    r"per_person",
    r"^pca_?\d*",
    r"^PC\d+$",
    r"^consumption$",           # the target itself
    r"^consumption_",           # old lag/rolling (wrong units, rolling includes the current row)
    r"^n_valid_readings$",      # known only after the hour ends
]


def find_forbidden(columns: Iterable[str]) -> List[str]:
    """Returns the columns matching any pattern in FORBIDDEN_PATTERNS."""
    rx = re.compile("|".join(f"(?:{p})" for p in FORBIDDEN_PATTERNS), flags=re.IGNORECASE)
    return [c for c in columns if rx.search(str(c))]


def assert_no_forbidden(columns: Iterable[str]) -> None:
    """Raises ValueError if the feature list contains a forbidden (leaking) column."""
    bad = find_forbidden(columns)
    if bad:
        raise ValueError(f"Forbidden features (target leakage): {bad}")
