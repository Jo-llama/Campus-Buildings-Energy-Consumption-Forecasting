import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

import warnings

warnings.filterwarnings('ignore')


class ModelBenchmark:
    def __init__(self):
        self.models = {
            'Lasso': Lasso(alpha=0.01, max_iter=1000),
            'Ridge': Ridge(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                                          random_state=42)
        }
        self.results = {}
        self.scaler = StandardScaler()
        self.trained_models = {}

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)

        # Scale once on train, transform test
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        for name, model in self.models.items():
            if name in ['Lasso', 'Ridge', 'ElasticNet']:
                Xtr, Xte = X_train_scaled, X_test_scaled
            else:
                Xtr, Xte = X_train.values, X_test.values

            model.fit(Xtr, y_train)

            y_train_pred = model.predict(Xtr)
            y_test_pred = model.predict(Xte)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            self.results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            self.trained_models[name] = model

            print(f"\n{name}")
            print(f"  Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
            print(f"  Train MAE:  {train_mae:.4f} | Test MAE:  {test_mae:.4f}")
            print(f"  Train R²:   {train_r2:.4f} | Test R²:   {test_r2:.4f}")

        return self.results

    def get_best_model(self):
        """Returns the best model according to test RMSE."""
        best_name = min(self.results, key=lambda x: self.results[x]['test_rmse'])
        return best_name, self.trained_models[best_name]

    def get_results_df(self):
        """Get results as a sorted DataFrame."""
        df = pd.DataFrame(self.results).T
        df = df.sort_values('test_rmse')
        return df


def get_feature_importance(model, feature_names, top_n=20):
    """Feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        return importance_df
    else:
        return None


def get_coefficients(model, feature_names, top_n=20):
    """Gets coefficients for linear models."""
    if hasattr(model, 'coef_'):
        coef = model.coef_
        indices = np.argsort(np.abs(coef))[::-1][:top_n]

        coef_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'coefficient': coef[indices]
        })
        return coef_df
    else:
        return None


def plot_learning_curves_trained_models(benchmark, X_train, y_train, X_test, y_test):
    """
    Learning curves using TimeSeriesSplit for already-trained models.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score

    tscv = TimeSeriesSplit(n_splits=5)
    models_to_plot = ['Lasso', 'GradientBoosting']

    fig, axes = plt.subplots(len(models_to_plot), 2, figsize=(15, 10))
    if len(models_to_plot) == 1:
        axes = axes.reshape(1, -1)

    for i, model_name in enumerate(models_to_plot):
        model = benchmark.trained_models[model_name]

        # Scaling for Lasso — reuse the scaler fitted in benchmark
        if model_name == 'Lasso':
            scaler = benchmark.scaler
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            # GradientBoosting does not require scaling
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        train_scores_rmse, val_scores_rmse = [], []
        train_scores_r2, val_scores_r2 = [], []

        # TimeSeriesSplit over training data
        for train_idx, val_idx in tscv.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            train_pred = model.predict(X_tr)
            val_pred = model.predict(X_val)

            train_scores_rmse.append(np.sqrt(mean_squared_error(y_tr, train_pred)))
            val_scores_rmse.append(np.sqrt(mean_squared_error(y_val, val_pred)))
            train_scores_r2.append(r2_score(y_tr, train_pred))
            val_scores_r2.append(r2_score(y_val, val_pred))

        # Results on test set
        test_pred = model.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)

        # RMSE plot
        axes[i, 0].plot(range(1, len(train_scores_rmse) + 1), train_scores_rmse, 'o-',
                        label='Train RMSE', color='blue', linewidth=2)
        axes[i, 0].plot(range(1, len(val_scores_rmse) + 1), val_scores_rmse, 'o-',
                        label='Validation RMSE', color='red', linewidth=2)
        axes[i, 0].axhline(y=test_rmse, color='green', linestyle='--', linewidth=2,
                           label=f'Test RMSE: {test_rmse:.2f}')
        axes[i, 0].set_title(f'{model_name} - RMSE')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        # R² plot
        axes[i, 1].plot(range(1, len(train_scores_r2) + 1), train_scores_r2, 'o-',
                        label='Train R²', color='blue', linewidth=2)
        axes[i, 1].plot(range(1, len(val_scores_r2) + 1), val_scores_r2, 'o-',
                        label='Validation R²', color='red', linewidth=2)
        axes[i, 1].axhline(y=test_r2, color='green', linestyle='--', linewidth=2,
                           label=f'Test R²: {test_r2:.3f}')
        axes[i, 1].set_title(f'{model_name} - R²')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def lasso_feature_selection(
        X_train,
        y_train,
        alpha=0.1,
        max_iter=10000,
        top_k=None,
        coef_threshold=1e-5
):
    """
    Feature selection using Lasso on training data.

    Returns:
    - selected_features: list of selected feature names
    - lasso_model: trained Lasso model (on scaled data)
    - scaler: StandardScaler fitted on X_train
    - coef_series: pandas Series with signed coefficient values
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
    lasso.fit(X_scaled, y_train)

    coefs = pd.Series(lasso.coef_, index=X_train.columns)

    # Selection:
    if top_k is not None:
        # Take top_k features by absolute coefficient value
        selected_features = coefs.abs().sort_values(ascending=False).head(top_k).index.tolist()
    else:
        # Take all features with |coef| above threshold
        selected_features = coefs[coefs.abs() > coef_threshold].index.tolist()

    return selected_features, lasso, scaler, coefs
