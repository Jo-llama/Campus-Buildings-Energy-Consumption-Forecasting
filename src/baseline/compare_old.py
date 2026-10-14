"""Krok 7.6: porównanie starego pipeline'u z uczciwym baseline'em (ablacja błędów).

Stare wyniki (results/modeling_results*.csv) nie są porównywalne wprost: dane 15-min
(kWh/15 min), inne wiersze (dropna usunęło liczniki i okres po 2021-06), inne zadanie.
Dlatego każdy błąd starego pipeline'u jest odtwarzany na danych godzinowych osobno,
a każdy wariant oceniany jest razem z naive_last na tych samych wierszach (skill).
"""
from typing import Dict, List, Tuple

import pandas as pd

from src.baseline import config
from src.baseline.features import feature_columns
from src.baseline.metrics import regression_metrics
from src.baseline.models import predict_lgbm, train_lgbm
from src.baseline.split import make_splits, required_columns

# Nazwy jak w starym pipeline (łapie je strażnik z leakage.py).
# Stare create_building_features: target / stała budynku -> target odtwarzalny wprost
LEAK_RATIO: List[str] = ["consumption_per_sqm", "consumption_per_person"]
# Stare create_rolling_features: okno obejmuje bieżący wiersz (y_t), min_periods=1
OLD_ROLL_WINDOWS: List[int] = [3, 6, 24]
LEAK_ROLL: List[str] = [f"consumption_rolling_{s}_{w}h" for w in OLD_ROLL_WINDOWS for s in ("mean", "std")]

# Warianty ablacji: nazwa -> (dodatkowe cechy, schemat splitu)
VARIANTS: Dict[str, Tuple[List[str], str]] = {
    "honest": ([], "time"),
    "+ratio_leak": (LEAK_RATIO, "time"),
    "+rolling_leak": (LEAK_ROLL, "time"),
    "+both_leaks": (LEAK_RATIO + LEAK_ROLL, "time"),
    "old_split": ([], "old"),
    "old_split+both_leaks": (LEAK_RATIO + LEAK_ROLL, "old"),
}


def add_old_leaky_features(f: pd.DataFrame) -> pd.DataFrame:
    """Dodaje cechy wyciekające jak w starym feature_engineering.py (na siatce godzinowej).

    Wejście: features_h{h}.parquet przed splitem (kompletna siatka per licznik).
    Zaślepki 0 w powierzchni/pojemności są już NaN (7.2), więc iloraz jest NaN jak w starym
    kodzie (replace(0, NA)).
    """
    f = f.sort_values(config.KEYS, kind="mergesort").copy()
    y = f[config.TARGET]
    f["consumption_per_sqm"] = y / f["gross_floor_area"]
    f["consumption_per_person"] = y / f["capacity"]
    g = y.groupby(f["meter_id"], sort=False)
    for w in OLD_ROLL_WINDOWS:
        f[f"consumption_rolling_mean_{w}h"] = g.transform(lambda s: s.rolling(w, min_periods=1).mean())
        f[f"consumption_rolling_std_{w}h"] = g.transform(lambda s: s.rolling(w, min_periods=1).std())
    return f


def old_positional_split(f: pd.DataFrame, horizon_h: int, test_size: float = 0.2,
                         val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
    """Stary split: iloc 80/20 na wierszach posortowanych po (meter_id, timestamp).

    W praktyce to split po licznikach (test = liczniki o najwyższych id, ten sam okres
    co train). Val (do early stopping) = ostatnie val_size wierszy części treningowej.
    Te same wymagane kolumny co w 7.4, żeby naive_last był zdefiniowany.
    """
    df = (f.dropna(subset=required_columns(horizon_h))
          .sort_values(["meter_id", "timestamp"], kind="mergesort").reset_index(drop=True))
    n_test = int(len(df) * (1 - test_size))
    n_val = int(n_test * (1 - val_size))
    return {"train": df.iloc[:n_val].reset_index(drop=True),
            "val": df.iloc[n_val:n_test].reset_index(drop=True),
            "test": df.iloc[n_test:].reset_index(drop=True)}


def time_split(f: pd.DataFrame, horizon_h: int) -> Dict[str, pd.DataFrame]:
    """Uczciwy split 7.4 (daty z config, przerwy GAP_DAYS)."""
    f = f.copy()
    f.attrs["horizon_h"] = horizon_h
    return make_splits(f, config.TRAIN_END, config.VAL_END, config.GAP_DAYS)


def evaluate_variant(splits: Dict[str, pd.DataFrame], horizon_h: int, extra: List[str],
                     **train_kwargs) -> Dict:
    """Uczy LightGBM z cechami honest + extra; metryki test razem z naive_last na tych wierszach."""
    feats = feature_columns(horizon_h) + extra
    booster = train_lgbm(splits["train"], splits["val"], horizon_h, features=feats,
                         allow_leaky=bool(extra), **train_kwargs)
    te = splits["test"]
    y = te[config.TARGET].to_numpy()
    lgbm = regression_metrics(y, predict_lgbm(booster, te, horizon_h).to_numpy())
    naive = regression_metrics(y, te[f"lag_{horizon_h}h"].to_numpy())
    span = (str(te["timestamp"].min()), str(te["timestamp"].max()))
    return {
        "lgbm": lgbm,
        "naive_last": naive,
        "skill_mae_vs_naive_last": 1 - lgbm["mae"] / naive["mae"],
        "best_iteration": int(booster.best_iteration),
        "test_meters": sorted(int(m) for m in te["meter_id"].unique()),
        "train_meters_in_test": int(te["meter_id"].isin(splits["train"]["meter_id"]).sum()),
        "test_span": span,
        "top_features": dict(sorted(zip(booster.feature_name(),
                                        booster.feature_importance("gain").round(0).tolist()),
                                    key=lambda kv: -kv[1])[:5]),
    }


def run_ablation(f: pd.DataFrame, horizon_h: int, variants: Dict = VARIANTS,
                 **train_kwargs) -> Dict[str, Dict]:
    """Uruchamia wszystkie warianty ablacji na cechach z add_old_leaky_features."""
    leaky = add_old_leaky_features(f)
    split_fns = {"time": time_split, "old": old_positional_split}
    cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    out: Dict[str, Dict] = {}
    for name, (extra, scheme) in variants.items():
        if scheme not in cache:
            cache[scheme] = split_fns[scheme](leaky, horizon_h)
        print(f"--- wariant {name} ({scheme} split, +{len(extra)} cech) ---")
        out[name] = {"split": scheme, "extra_features": extra,
                     **evaluate_variant(cache[scheme], horizon_h, extra, **train_kwargs)}
    return out


def ablation_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Tabela: wiersz = wariant; MAE/R2 modelu, MAE naive_last na tych samych wierszach, skill."""
    rows = {name: {
        "split": r["split"],
        "n_test": r["lgbm"]["n"],
        "mae": r["lgbm"]["mae"],
        "rmse": r["lgbm"]["rmse"],
        "r2": r["lgbm"]["r2"],
        "wape_pct": r["lgbm"]["wape_pct"],
        "naive_last_mae": r["naive_last"]["mae"],
        "skill_mae_vs_naive_last": r["skill_mae_vs_naive_last"],
    } for name, r in results.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def old_reported_vs_persistence(results_csvs: List, features_csv, test_size: float = 0.2) -> Dict:
    """Stare raportowane metryki vs persistence (lag 1 wiersza = 15 min) na starym teście.

    Stary test = ostatnie 20% wierszy features_data.csv (kolejność jak features_data_encoded,
    sort po meter_id). consumption_lag_1h w starym pliku to w praktyce lag 15 min.
    """
    f = pd.read_csv(features_csv, usecols=["meter_id", "timestamp", "consumption",
                                           "consumption_lag_1h"], parse_dates=["timestamp"])
    te = f.iloc[int(len(f) * (1 - test_size)):]
    tr = f.iloc[:int(len(f) * (1 - test_size))]
    naive = regression_metrics(te["consumption"].to_numpy(), te["consumption_lag_1h"].to_numpy())
    reported = {}
    for path in results_csvs:
        r = pd.read_csv(path, index_col=0)
        reported[path.name] = {m: {"test_mae": float(row["test_mae"]), "test_r2": float(row["test_r2"]),
                                   "skill_mae_vs_persistence_15min": float(1 - row["test_mae"] / naive["mae"])}
                               for m, row in r.iterrows()}
    return {
        "old_rows": int(len(f)),
        "old_meters": int(f["meter_id"].nunique()),
        "old_span": [str(f["timestamp"].min()), str(f["timestamp"].max())],
        "old_test_meters": sorted(int(m) for m in te["meter_id"].unique()),
        "old_test_meters_also_in_train": sorted(int(m) for m in set(te["meter_id"]) & set(tr["meter_id"])),
        "old_test_span": [str(te["timestamp"].min()), str(te["timestamp"].max())],
        "persistence_15min_on_old_test": naive,
        "reported": reported,
    }
