import numpy as np
import optuna
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib


def tune_lasso_optuna(X_train_fs, X_test_fs, y_train, y_test, n_trials=30,
                      save_path_model="models/best_lasso_optuna.joblib",
                      save_path_scaler="models/best_lasso_scaler.joblib",
                      save_study_path=None):
    def objective_lasso(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        max_iter = trial.suggest_int("max_iter", 2000, 20000)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fs)
        X_test_scaled = scaler.transform(X_test_fs)

        model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return rmse

    study = optuna.create_study(direction="minimize", study_name="lasso_optuna")
    study.optimize(objective_lasso, n_trials=n_trials)

    best_alpha = study.best_params["alpha"]
    best_max_iter = study.best_params["max_iter"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fs)
    X_test_scaled = scaler.transform(X_test_fs)

    best_lasso = Lasso(alpha=best_alpha, max_iter=best_max_iter, random_state=42)
    best_lasso.fit(X_train_scaled, y_train)

    # === ZAPIS MODELU I SCALERA ===
    if save_path_model is not None:
        joblib.dump(best_lasso, save_path_model)
    if save_path_scaler is not None:
        joblib.dump(scaler, save_path_scaler)
    if save_study_path is not None:
        joblib.dump(study, save_study_path)

    return study, best_lasso, scaler


def tune_xgb_optuna(
    X_train_fs,
    X_test_fs,
    y_train,
    y_test,
    n_trials=30,
    save_path_model="models/best_xgb_optuna.joblib",
    save_study_path=None,
):
    """Optuna dla XGBoost na zredukowanych cechach + zapis najlepszego modelu."""

    def objective_xgb(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }

        model = xgb.XGBRegressor(
            **params,
            tree_method="hist",
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train_fs, y_train)
        preds = model.predict(X_test_fs)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return rmse

    study = optuna.create_study(direction="minimize", study_name="xgboost_optuna")
    study.optimize(objective_xgb, n_trials=n_trials)

    best_params = study.best_params

    best_xgb = xgb.XGBRegressor(
        **best_params,
        tree_method="hist",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )
    best_xgb.fit(X_train_fs, y_train)

    # === ZAPIS NAJLEPSZEGO MODELU ===
    if save_path_model is not None:
        joblib.dump(best_xgb, save_path_model)
    if save_study_path is not None:
        joblib.dump(study, save_study_path)

    return study, best_xgb
