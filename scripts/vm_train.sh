#!/usr/bin/env bash
# Runs ON the Azure VM (Ubuntu 22.04): installs deps, clones the repo, fetches the
# UNICON data from Kaggle and runs the honest-baseline pipeline for each horizon.
# Normally started by `scripts/azure_vm.sh train`; safe to re-run (finished steps
# are skipped when their outputs already exist).
set -euo pipefail

REPO="${REPO:-https://github.com/Jo-llama/Campus-Buildings-Energy-Consumption-Forecasting.git}"
BRANCH="${BRANCH:-honest-baseline}"
HORIZONS="${HORIZONS:-1 24}"
DIR="$HOME/project"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "Installing system packages"
# Finish any install cut off by a shutdown, otherwise apt refuses to run
sudo dpkg --configure -a
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git python3-venv python3-pip libgomp1

if [[ -d "$DIR/.git" ]]; then
  log "Updating repo"
  git -C "$DIR" fetch -q origin "$BRANCH"
  git -C "$DIR" checkout -q "$BRANCH"
  git -C "$DIR" pull -q --ff-only
else
  log "Cloning $REPO ($BRANCH)"
  git clone -q -b "$BRANCH" "$REPO" "$DIR"
fi
cd "$DIR"

log "Setting up virtualenv"
[[ -d .venv ]] || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

# hourly.parquet is built from merged_data.csv, which is rebuilt from data/raw;
# data/raw is downloaded from Kaggle on first use (src/baseline/io.py).
if [[ ! -f data/processed/hourly.parquet ]]; then
  log "Step resample (downloads UNICON from Kaggle, merges, resamples to hourly)"
  python -m src.baseline.run --step resample
else
  log "hourly.parquet exists, skipping resample"
fi

# After resample, so the data-dependent tests (leakage, resample) run instead of skipping
log "Running tests"
python -m pytest -q tests

for h in $HORIZONS; do
  # split + train share data/processed/splits, so each horizon runs features -> split -> train in turn
  log "Horizon h=$h: features"
  python -m src.baseline.run --step features --horizon "$h"
  log "Horizon h=$h: split"
  python -m src.baseline.run --step split --horizon "$h"
  log "Horizon h=$h: train"
  python -m src.baseline.run --step train --horizon "$h"
done

log "Done. Outputs in $DIR/results and $DIR/models"
