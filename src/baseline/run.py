"""CLI uczciwego baseline'u: python -m src.baseline.run --step <krok>."""
import argparse
import json
from typing import Dict

import pandas as pd

from src.baseline import config
from src.baseline.compare_old import ablation_table, old_reported_vs_persistence, run_ablation
from src.baseline.features import (STATIC_NUMERIC, add_static_features, build_features,
                                    feature_columns)
from src.baseline.io import load_calendar, load_events, load_merged, save_parquet
from src.baseline.leakage import find_forbidden
from src.baseline.metrics import metrics_table, per_meter_metrics, skill_vs
from src.baseline.models import (all_predictions, feature_importance, model_names,
                                  naive_columns, train_lgbm)
from src.baseline.resample import hourly_summary, to_hourly
from src.baseline.split import coverage_by_meter, make_splits


def step_resample() -> Dict:
    """Krok 7.1: merged_data.csv -> hourly.parquet (+ events.parquet); zwraca raport."""
    df = load_merged()
    hourly, report = to_hourly(df, calendar=load_calendar())
    save_parquet(hourly, config.HOURLY_PARQUET)
    save_parquet(load_events(), config.EVENTS_PARQUET)

    summary = hourly_summary(hourly)
    report["meters"] = int(hourly["meter_id"].nunique())
    report["date_range"] = [str(hourly["timestamp"].min()), str(hourly["timestamp"].max())]
    report["nan_target_pct_total"] = round(100 * hourly[config.TARGET].isna().mean(), 2)
    report["nan_pct_exog"] = (100 * hourly[config.WEATHER_COLS + config.CALENDAR_COLS]
                              .isna().mean()).round(2).to_dict()
    print(json.dumps(report, indent=2))
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(summary.sort_values("nan_target_pct", ascending=False))
    return report


def step_static() -> Dict:
    """Krok 7.2: sprawdza brak cech wyciekających i raportuje cechy statyczne budynku."""
    hourly = pd.read_parquet(config.HOURLY_PARQUET)
    old = pd.read_csv(config.DATA_PROCESSED / "features_data_pca.csv", nrows=0).columns
    df, static_report = add_static_features(hourly)
    report = {
        "rows_before": int(len(hourly)),
        "rows_after": int(len(df)),
        "old_pipeline_forbidden_cols": find_forbidden(old),
        "hourly_forbidden_cols": find_forbidden(c for c in hourly.columns),
        "static": static_report,
        "row_missing_pct": (100 * df[STATIC_NUMERIC].isna().mean()).round(2).to_dict(),
    }
    print(json.dumps(report, indent=2))
    return report


def step_features(horizon_h: int = config.HORIZON_H) -> Dict:
    """Krok 7.3: hourly.parquet -> features_h{h}.parquet; raport braków i korelacji."""
    hourly = pd.read_parquet(config.HOURLY_PARQUET)
    f = build_features(hourly, horizon_h)
    path = config.DATA_PROCESSED / config.FEATURES_PARQUET_TMPL.format(h=horizon_h)
    save_parquet(f, path)
    feats = feature_columns(horizon_h)
    sample = f.dropna(subset=[config.TARGET]).sample(300_000, random_state=config.SEED)
    num = [c for c in feats if c != "category"]
    rho = sample[num].corrwith(sample[config.TARGET], method="spearman").abs()
    report = {
        "path": str(path),
        "rows": int(len(f)),
        "n_features": len(feats),
        "features": feats,
        "nan_pct": (100 * f[feats].isna().mean()).round(2)[lambda x: x > 0].to_dict(),
        "spearman_abs_top5": rho.sort_values(ascending=False).head(5).round(4).to_dict(),
    }
    print(json.dumps(report, indent=2))
    return report


def step_split(horizon_h: int = config.HORIZON_H) -> Dict:
    """Krok 7.4: features_h{h}.parquet -> splits/{train,val,test}.parquet + manifest.json."""
    f = pd.read_parquet(config.DATA_PROCESSED / config.FEATURES_PARQUET_TMPL.format(h=horizon_h))
    f.attrs["horizon_h"] = horizon_h
    splits = make_splits(f, config.TRAIN_END, config.VAL_END, config.GAP_DAYS)
    for name, part in splits.items():
        save_parquet(part, config.SPLITS_DIR / f"{name}.parquet")
    cov = coverage_by_meter(splits)
    manifest = {
        "horizon_h": horizon_h,
        "train_end": config.TRAIN_END, "val_end": config.VAL_END, "gap_days": config.GAP_DAYS,
        "splits": {n: s.attrs["report"] for n, s in splits.items()},
        "rows_per_meter": {n: {int(m): int(v) for m, v in cov[n].items()} for n in cov.columns},
    }
    (config.SPLITS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows_per_meter"}, indent=2))
    thin = cov[(cov["val"] < 500) | (cov["test"] < 500)]
    print(f"Liczniki z < 500 wierszy w val lub test ({len(thin)}):")
    print(thin.to_string())
    return manifest


def step_train(horizon_h: int = config.HORIZON_H) -> Dict:
    """Krok 7.5: splits -> naive + LightGBM; metryki val/test, predykcje, model, ważności."""
    manifest = json.loads((config.SPLITS_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["horizon_h"] != horizon_h:
        raise ValueError(f"Splity zbudowane dla h={manifest['horizon_h']}, a nie h={horizon_h} "
                         f"- uruchom --step features/split --horizon {horizon_h}")
    splits = {n: pd.read_parquet(config.SPLITS_DIR / f"{n}.parquet") for n in ("train", "val", "test")}
    booster = train_lgbm(splits["train"], splits["val"], horizon_h)

    tag = f"h{horizon_h}"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(config.MODELS_DIR / f"lgbm_{tag}.txt"), num_iteration=booster.best_iteration)
    feature_importance(booster).to_csv(config.RESULTS_DIR / f"feature_importance_{tag}.csv", index=False)

    names = model_names(horizon_h)
    best_naive, per_meter = {}, {}
    report: Dict = {"horizon_h": horizon_h, "best_iteration": int(booster.best_iteration),
                    "naive_columns": naive_columns(horizon_h), "params": config.LGBM_PARAMS,
                    "metrics": {}}
    for split in ("val", "test"):
        pred = all_predictions(booster, splits[split], horizon_h)
        save_parquet(pred, config.RESULTS_DIR / f"predictions_{tag}_{split}.parquet")
        table = metrics_table(pred, names)
        # Najlepszy naive wybierany na val i używany jako punkt odniesienia także na test
        if split == "val":
            best_naive["name"] = table.loc[list(naive_columns(horizon_h)), "mae"].idxmin()
        table["skill_mae_vs_best_naive"] = skill_vs(table, best_naive["name"])
        report["metrics"][split] = table.round(4).to_dict(orient="index")
        pm = per_meter[split] = per_meter_metrics(pred, names)
        pm.to_csv(config.RESULTS_DIR / f"per_meter_{tag}_{split}.csv", index=False)
        print(f"\n=== {split} (n={len(pred)}) ===")
        print(table.round(3).to_string())
    report["best_naive"] = best_naive["name"]

    # Per licznik na test: ile liczników lgbm bije najlepszy naive
    wide = per_meter["test"].pivot(index="meter_id", columns="model", values="mae")
    beats = wide["lgbm"] < wide[best_naive["name"]]
    report["test_meters_lgbm_beats_best_naive"] = f"{int(beats.sum())}/{len(beats)}"
    report["test_meters_lgbm_worse"] = [int(m) for m in wide.index[~beats]]
    (config.RESULTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(report, indent=2),
                                                           encoding="utf-8")
    print(f"\nbest_iteration={booster.best_iteration}, najlepszy naive (val): {best_naive['name']}")
    print(f"LightGBM lepszy od {best_naive['name']} na test w {report['test_meters_lgbm_beats_best_naive']} licznikach")
    return report


def step_compare_old(horizon_h: int = config.HORIZON_H) -> Dict:
    """Krok 7.6: ablacja błędów starego pipeline'u + stare wyniki vs persistence 15 min."""
    f = pd.read_parquet(config.DATA_PROCESSED / config.FEATURES_PARQUET_TMPL.format(h=horizon_h))
    results = run_ablation(f, horizon_h)
    table = ablation_table(results)
    old_csvs = sorted((config.PROJECT_ROOT / "results").glob("modeling_results*.csv"))
    old = old_reported_vs_persistence(old_csvs, config.DATA_PROCESSED / "features_data.csv")

    tag = f"h{horizon_h}"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.RESULTS_DIR / f"compare_old_ablation_{tag}.csv")
    report = {"horizon_h": horizon_h, "ablation": results, "old_pipeline": old}
    (config.RESULTS_DIR / f"compare_old_{tag}.json").write_text(json.dumps(report, indent=2),
                                                               encoding="utf-8")
    with pd.option_context("display.width", 200):
        print("\n=== Ablacja (test, LightGBM vs naive_last na tych samych wierszach) ===")
        print(table.round(4).to_string())
    print("\n=== Stary pipeline: raportowane metryki vs persistence 15 min na starym teście ===")
    print(json.dumps({k: v for k, v in old.items() if k != "reported"}, indent=2))
    for name, models in old["reported"].items():
        print(name)
        print(pd.DataFrame.from_dict(models, orient="index").round(4).to_string())
    return report


STEPS = {"resample": step_resample, "static": step_static, "features": step_features,
         "split": step_split, "train": step_train, "compare_old": step_compare_old}


def main() -> None:
    """Parsuje argumenty i uruchamia wybrany krok."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, choices=sorted(STEPS))
    parser.add_argument("--horizon", type=int, default=config.HORIZON_H,
                        help="horyzont prognozy w godzinach (--step features/split/train/compare_old)")
    args = parser.parse_args()
    if args.step in ("features", "split", "train", "compare_old"):
        STEPS[args.step](args.horizon)
    else:
        STEPS[args.step]()


if __name__ == "__main__":
    main()
