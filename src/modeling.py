import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class ModelBenchmark:
    """Porównanie wielu modeli."""

    def __init__(self):
        self.models = {
            'Lasso': Lasso(alpha=0.1, max_iter=10000),
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                                          random_state=42)
        }
        self.results = {}
        self.scaler = StandardScaler()
        self.trained_models = {}

    def prepare_data(self, X_train, X_test, y_train, y_test, model_name=None):
        """
        Skalowanie tylko dla modeli liniowych.
        """
        linear_models = ['Lasso', 'Ridge', 'ElasticNet']
        if model_name in linear_models:
            print(f"Skalowanie danych dla modelu: {model_name}")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, y_train, y_test
        else:
            print(f"{model_name} bez skalowania")
            return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Trening i ewaluacja wszystkich modeli."""
        print("\n" + "=" * 80)
        print("TRENING MODELI")
        print("=" * 80)

        X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data(
            X_train, X_test, y_train, y_test
        )

        for name, model in self.models.items():
            X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data(
                X_train, X_test, y_train, y_test, model_name=name
            )

            # Trening
            model.fit(X_train_scaled, y_train)

            # Predykcje
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Metryki
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

            print(f"  Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
            print(f"  Train MAE:  {train_mae:.4f} | Test MAE:  {test_mae:.4f}")
            print(f"  Train R²:   {train_r2:.4f} | Test R²:   {test_r2:.4f}")

        return self.results

    def get_best_model(self):
        """Zwraca najlepszy model na podstawie test RMSE."""
        best_name = min(self.results, key=lambda x: self.results[x]['test_rmse'])
        return best_name, self.trained_models[best_name]

    def get_results_df(self):
        """Zwraca wyniki jako DataFrame."""
        df = pd.DataFrame(self.results).T
        df = df.sort_values('test_rmse')
        return df


def get_feature_importance(model, feature_names, top_n=20):
    """Pobiera ważność cech dla modeli tree-based."""
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
    """Pobiera współczynniki dla modeli liniowych."""
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
