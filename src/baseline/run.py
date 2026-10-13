"""CLI uczciwego baseline'u: python -m src.baseline.run --step <krok>."""
import argparse
import json
from typing import Dict

import pandas as pd

from src.baseline import config
from src.baseline.features import STATIC_NUMERIC, add_static_features
from src.baseline.io import load_calendar, load_events, load_merged, save_parquet
from src.baseline.leakage import find_forbidden
from src.baseline.resample import hourly_summary, to_hourly


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


STEPS = {"resample": step_resample, "static": step_static}


def main() -> None:
    """Parsuje argumenty i uruchamia wybrany krok."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, choices=sorted(STEPS))
    args = parser.parse_args()
    STEPS[args.step]()


if __name__ == "__main__":
    main()
