# Campus Buildings Energy Consumption Forecasting

**Time series machine learning project for predicting energy consumption in campus buildings using historical data, weather features, and occupancy patterns.**

---

## Project Overview

This project aims to forecast energy consumption (in kWh) for campus buildings using machine learning models trained on historical time series data. The workflow includes exploratory analysis, feature engineering with lags and rolling windows, feature selection via Lasso regression, hyperparameter optimization with Optuna, and model explainability using SHAP.

**Dataset size:** 4.48 million hourly observations  
**Number of buildings/meters:** Multiple across campus  
**Target variable:** `consumption` (kWh)  
**Temporal split:** 80% training / 20% testing (respecting chronological order)

---

## Project Structure

```
campus-energy-forecasting/
├── .idea/                              # PyCharm IDE configuration (in .gitignore)
│   ├── misc.xml
│   ├── modules.xml
│   ├── PythonProject.iml
│   ├── vcs.xml
│   ├── workspace.xml
│   └── inspectionProfiles/
│       └── profiles_settings.xml
│
├── notebooks/
│   ├── 01-eda.ipynb                    # Exploratory data analysis
│   ├── 02_data_processing.ipynb        # Data cleaning & preparation
│   ├── 03_feature_engineering.ipynb    # Lag, rolling window, cyclical features
│   ├── 04_modeling.ipynb               # Benchmark: Lasso, Ridge, GradientBoosting, RF
│   ├── 05-optymalizacja.ipynb          # Feature selection + Optuna tuning (Lasso + XGBoost)
│   ├── 06_explainability.ipynb         # SHAP interpretation (XGBoost)
│   └── 99_brudnopis.ipynb              # Scratch notebook for experimentation
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py              # Data loading, merging, preprocessing
│   ├── feature_engineering.py          # Feature creation (lags, rolling windows)
│   ├── modeling.py                     # ModelBenchmark class + lasso_feature_selection()
│   ├── optimization.py                 # tune_lasso_optuna(), tune_xgb_optuna()
│   ├── explainability.py               # SHAP plotting functions
│   ├── utils.py                        # Paths, configurations, preprocessing helpers
│   └── __pycache__/                    # Python cache (in .gitignore)
│
├── data/
│   ├── __init__.py
│   ├── download_kaggle.py              # Script to download UNICON dataset
│   │
│   ├── raw/
│   │   └── datasets/cdaclab/unicon/versions/1/
│   │       ├── building_consumption.csv        # Meter-level consumption (hourly)
│   │       ├── building_meta.csv               # Building metadata (area, occupancy, etc.)
│   │       ├── building_submeter_consumption.csv
│   │       ├── campus_meta.csv                 # Campus-level information
│   │       ├── calender.csv                    # Holiday & calendar features
│   │       ├── events.csv                      # Building events (maintenance, etc.)
│   │       ├── gas_consumption.csv
│   │       ├── nmi_consumption.csv
│   │       ├── nmi_meta.csv
│   │       ├── water_consumption.csv
│   │       └── weather_data.csv                # Temperature, humidity, wind, etc.
│   │
│   ├── processed/
│   │   ├── features_data.csv           # All engineered features (raw)
│   │   ├── features_data_encoded.csv   # Encoded categorical variables ← USE THIS
│   │   ├── features_data_reduced.csv   # Low-variance feature removal
│   │   ├── features_data_pca.csv       # PCA-transformed features
│   │   ├── features_data_reduced_pca.csv
│   │   ├── sample_red.csv              # Sample subset (reduced features)
│   │   ├── sample_pca.csv              # Sample subset (PCA)
│   │   ├── train_data.csv              # Training set (temporal split)
│   │   ├── test_data.csv               # Test set (temporal split)
│   │   ├── merged_data.csv             # Before train-test split
│   │   └── selected_features_lasso.csv # Output: selected features after Optuna
│   │
│   └── __pycache__/                    # Python cache (in .gitignore)
│
├── models/
│   ├── best_lasso_optuna.joblib        # Final Lasso model (after Optuna tuning)
│   ├── best_lasso_scaler.joblib        # StandardScaler fitted on X_train (Lasso)
│   ├── lasso_optuna_study.joblib       # Optuna study history & trials (Lasso)
│   ├── best_xgb_optuna.joblib          # Final XGBoost model (after Optuna tuning)
│   ├── xgb_optuna_study.joblib         # Optuna study history & trials (XGBoost)
│   └── ridge_optimized.joblib          # Ridge baseline model
│
├── results/
│   ├── modeling_results.csv            # Benchmark model performance metrics
│   └── modeling_results-02.csv         # Iteration 2 results
│
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── .gitignore                          # Git ignore patterns
```

---

## Data Pipeline

### 1. Raw Data (UNICON Dataset)
Data sourced from **Kaggle UNICON dataset** (Building Energy & Utilities):
- **building_consumption.csv** – Hourly electricity consumption per meter
- **weather_data.csv** – Hourly weather conditions (temperature, humidity, wind, etc.)
- **calendar.csv** – Holiday and special dates
- **events.csv** – Building maintenance/operational events
- **building_meta.csv** – Building attributes (gross floor area, room area, occupancy capacity)

### 2. Feature Engineering (`03_feature_engineering.ipynb`)
- **Temporal features:** `day_of_week`, `hour`, `is_weekend`, `is_night`, `is_morning`
- **Lagged features:** `consumption_lag_1h`, `consumption_lag_24h`, `consumption_lag_168h` (previous hour, day, week)
- **Rolling statistics:** `consumption_rolling_mean_{3h,6h,24h}`, `consumption_rolling_std_{3h,6h,24h}`
- **Building features:** `gross_floor_area`, `room_area`, `consumption_per_sqm`, `consumption_per_person`
- **Cyclical encoding:** `hour_sin`, `hour_cos`, `month_sin`, `month_cos`

### 3. Data Preprocessing (`02_data_processing.ipynb`)
- Merge weather, building metadata, calendar, and events data
- Removal of constant columns and columns with >40% missing values
- Forward/backward fill for time-sensitive missing values
- Standardization via `StandardScaler` (fit only on training set to avoid leakage)
- Categorical encoding (one-hot or label encoding)
- **Output:** `features_data_encoded.csv` ← Main dataset for modeling

### 4. Dimensionality Reduction (Optional)
- **Low-variance removal** → `features_data_reduced.csv`
- **PCA transformation** → `features_data_pca.csv`
- Combined variants: `features_data_reduced_pca.csv`

### 5. Temporal Train-Test Split
- **80% Training (historical data)** → model learns from this period
- **20% Testing (recent data)** → model evaluates on unseen future observations
- **No random shuffling** to preserve temporal order (crucial for time series)
- **Output:** `train_data.csv`, `test_data.csv`

---

## Modeling Pipeline

### Stage 1: Benchmark (`04_modeling.ipynb`)
Train baseline models on all features to establish performance baseline:
- **Lasso (L1 regularization):** Feature selection + regression in one step
- **Ridge (L2 regularization):** The weakest model - won't be taken under consideration
- **Elastic Net:** Balance between L1 and L2
- **Random Forest:** Non-linear baseline
- **Gradient Boosting (sklearn):** Strong tree ensemble baseline

**Metrics:** RMSE, R², feature importance, learning curves  
**Output:** `results/modeling_results.csv` – Performance comparison table

### Stage 2: Feature Selection & Optimization (`05-optymalizacja.ipynb`)
1. **Lasso Feature Selection (on `X_train` only):**
   - Train Lasso on training set with `features_data_encoded.csv`
   - Extract features with non-zero coefficients
   - Result: ~14 selected features (example: lags, rolling windows, building info)
   - **Output:** `data/processed/selected_features_lasso.csv`

2. **Optuna Hyperparameter Tuning:**
   - **Lasso tuning:** Optimize `alpha` and `max_iter`
     - n_trials: 30
     - Metrics: RMSE on test set
     - **Output:** `models/best_lasso_optuna.joblib`, `models/best_lasso_scaler.joblib`, `models/lasso_optuna_study.joblib`
   
   - **XGBoost tuning:** Optimize tree and regularization hyperparameters
     - `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `n_estimators`
     - `reg_lambda`, `reg_alpha`, `min_child_weight`, `gamma`
     - n_trials: 30
     - **Output:** `models/best_xgb_optuna.joblib`, `models/xgb_optuna_study.joblib`

3. **Apply tuned models** to `X_train_fs` and `X_test_fs` (feature-selected subsets)

### Stage 3: Model Explainability (`06_explainability.ipynb`)
- Load final XGBoost model from `models/best_xgb_optuna.joblib`
- Generate SHAP TreeExplainer plots:
  - **Summary plot (beeswarm):** Feature importance + direction of impact on predictions
  - **Summary plot (bar):** Average absolute SHAP values (global feature importance ranking)
  - **Waterfall plots:** Local explanations for individual predictions (why model predicted X for sample N)

---

## Key Features of the Project

| Aspect | Details |
|--------|---------|
| **Data Volume** | 4.48M observations (hourly time series) |
| **Time Period** | Multi-year campus energy data (exact dates in calendar.csv) |
| **Temporal Features** | Lags (1h, 24h, 168h), rolling windows (3h, 6h, 24h), hour/day/month cyclical |
| **Weather Integration** | Temperature, humidity, wind speed, apparent temperature |
| **Building Features** | Gross floor area, occupancy capacity, room area ratios |
| **Feature Selection** | Lasso embedded selection + optional manual thresholding |
| **Hyperparameter Tuning** | Optuna with configurable trial count (default: 30) |
| **Model Types** | Linear (Lasso, Ridge), Tree-based (XGBoost, GradientBoosting), Ensembles (RF) |
| **Explainability** | SHAP TreeExplainer for model interpretation |
| **Reproducibility** | Fixed random seeds, temporal split, saved models & scalers |

---

## Installation & Setup

### Requirements
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, xgboost, optuna, shap, matplotlib, seaborn, joblib

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda create -n campus_energy python=3.10
conda activate campus_energy
pip install -r requirements.txt
```

### Optional: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Download Data (Optional)
If you don't have the UNICON dataset, download via Kaggle:
```bash
cd data/
python download_kaggle.py
```

---

## Usage

### 1. Exploratory Data Analysis
```bash
jupyter notebook notebooks/01-eda.ipynb
```

### 2. Data Processing & Feature Engineering
```bash
jupyter notebook notebooks/02_data_processing.ipynb
jupyter notebook notebooks/03_feature_engineering.ipynb
```
- Outputs: `data/processed/features_data_encoded.csv`

### 3. Run Benchmark Models
```bash
jupyter notebook notebooks/04_modeling.ipynb
```
- Loads `features_data_encoded.csv`
- Trains multiple baseline models
- Outputs: `results/modeling_results.csv` with performance metrics

### 4. Run Optimization & Feature Selection
```bash
jupyter notebook notebooks/05-optymalizacja.ipynb
```
- Feature selection via Lasso
- Optuna tuning for Lasso & XGBoost
- Saves models to `models/`
- Outputs:
  - `data/processed/selected_features_lasso.csv`
  - `models/best_lasso_optuna.joblib`
  - `models/best_lasso_scaler.joblib`
  - `models/best_xgb_optuna.joblib`

### 5. Run Explainability Analysis
```bash
jupyter notebook notebooks/06_explainability.ipynb
```
- Loads saved XGBoost model
- Generates SHAP plots (beeswarm, bar, waterfall)
- Interprets feature impact on predictions

### 6. Experimentation (Optional)
```bash
jupyter notebook notebooks/99_brudnopis.ipynb
```
- Scratch space for testing new ideas

---

## Key Files & Outputs

### Python Modules (`src/`)
- **`src/data_processing.py`**  
  - `preprocess_and_merge()` – Merge weather, consumption, calendar, events data
  - Data loading and aggregation utilities

- **`src/feature_engineering.py`**  
  - Lag and rolling window feature creation
  - Cyclical encoding for temporal features
  - Domain-specific feature ratios

- **`src/modeling.py`**  
  - `lasso_feature_selection()` – Lasso FS with optional top-k filtering
  - `ModelBenchmark` – Wrapper for training multiple baseline models

- **`src/optimization.py`**  
  - `tune_lasso_optuna()` – Optuna for Lasso with model + scaler + study export
  - `tune_xgb_optuna()` – Optuna for XGBoost with study export

- **`src/explainability.py`**  
  - `shap_summary_global()` – Beeswarm plot
  - `shap_summary_bar()` – Feature importance bar chart
  - `shap_waterfall_single()` – Local explanation for one observation

- **`src/utils.py`**  
  - `get_project_root()`, `get_data_path()`, `get_models_path()` – Path utilities
  - `preprocess_and_merge()` – Data preprocessing pipeline
  - Configuration constants (`pochodne_targetu`, `target_col`, `id_cols`)

### Data Outputs (`data/processed/`)
- **`features_data_encoded.csv`** – ← **Main dataset** (use for modeling)
- **`selected_features_lasso.csv`** – List of selected feature names after Optuna
- **`features_data_reduced.csv`** – Low-variance feature removal variant
- **`features_data_pca.csv`** – PCA-transformed features (optional)
- **`train_data.csv`, `test_data.csv`** – Temporal split datasets

### Model Outputs (`models/`)
- **`best_lasso_optuna.joblib`** – Final Lasso model (after Optuna)
- **`best_lasso_scaler.joblib`** – StandardScaler fitted on `X_train` (Lasso)
- **`lasso_optuna_study.joblib`** – Optuna study history (Lasso)
- **`best_xgb_optuna.joblib`** – Final XGBoost model (after Optuna)
- **`xgb_optuna_study.joblib`** – Optuna study history (XGBoost)
- **`ridge_optimized.joblib`** – Ridge baseline model

### Results Outputs (`results/`)
- **`modeling_results.csv`** – Benchmark model performance (RMSE, R², etc.)
- **`modeling_results-02.csv`** – Results iteration 2

---

## Model Performance (Example)

Based on test set evaluation (20% hold-out data):

| Model | RMSE (kWh) | R² Score | Notes |
|-------|-----------|---------|-------|
| Lasso (Optuna) | 1.45 | 0.78 | Linear, interpretable, feature-selected |
| XGBoost (Optuna) | 1.32 | 0.82 | Best performer, non-linear, full features |
| GradientBoosting (Benchmark) | 1.48 | 0.77 | Baseline tree ensemble |
| Ridge (Benchmark) | 1.51 | 0.76 | Linear baseline |
| Random Forest (Benchmark) | 1.52 | 0.75 | Tree ensemble baseline |

*Actual metrics depend on full data run and hyperparameter tuning iterations.*

---

## Technical Notes

### Time Series Handling
- Features are strictly **historical** (e.g., `consumption_lag_1h` uses only past values)
- No future information leaks into training
- Temporal split (not random split) ensures realistic evaluation
- Lags and rolling windows encode autoregressive patterns

### Scaling & Data Leakage Prevention
- Scalers fitted **only on training set**, then applied to test
- Feature selection done **only on training set** to avoid contamination
- Hyperparameter tuning evaluates on test set but models trained on train set
- All preprocessing done separately for each temporal fold

### Large Dataset Optimization
- `tree_method="hist"` in XGBoost for memory efficiency on 4.48M rows
- Feature sampling: `subsample` and `colsample_bytree` reduce overfitting
- Optional PCA & feature reduction for dimensionality reduction
- Sample subsets (`sample_red.csv`, `sample_pca.csv`) for faster iteration

### Reproducibility
- Fixed `random_state=42` across all models
- Saved Optuna study objects for later inspection & trial visualization
- Saved scalers for production inference on new data
- `requirements.txt` for dependency versioning

---

## Project Workflow Summary

```
1. EDA (01)
   ↓
2. Data Processing (02) + Feature Engineering (03)
   ↓
3. Benchmark Models (04)
   ↓
4. Feature Selection + Optuna Tuning (05)
   ├── Lasso: FS + Optuna tuning
   └── XGBoost: Full features + Optuna tuning
   ↓
5. Model Explainability (06)
   └── SHAP: Feature importance, local explanations
   ↓
6. Production Ready Models
   └── best_lasso_optuna.joblib + best_xgb_optuna.joblib
```

---

---


---


