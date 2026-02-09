import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score
# PUNCC might be used in notebooks, but MAPIE is the standard maintained library.
# Implementing MAPIE based logic as seen in notebooks.

from uq_regression.utils import save_plot_to_excel, save_values_to_excel

# Wrapper for TabNet to be compatible with sklearn/MAPIE
class TabNetRegressorCP(TabNetRegressor):
    def fit(self, X, y, **kwargs):
        # MAPIE passes X, y as arrays. TabNet expects y to be 2D sometimes or specific format
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        # Add default eval_set if not provided to avoid errors if strict
        if 'eval_set' not in kwargs:
             # Just use training data for eval to prevent crashing, or split internally
             # For simplicity, we just pass X, y
             kwargs['eval_set'] = [(X, y)]
             kwargs['patience'] = 10
             kwargs['max_epochs'] = 50
             kwargs['batch_size'] = 1024
             kwargs['virtual_batch_size'] = 128
        super().fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return super().predict(X).flatten()

def conformal_prediction_analysis(X_train, y_train, X_test, y_test, output_dir):
    """
    Runs Conformal Prediction analysis using MAPIE (Standard CP) and adaptive methods if available.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    excel_path = os.path.join(output_dir, "conformal_results.xlsx")

    # Models to test
    base_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'TabNet': TabNetRegressorCP(verbose=0, seed=42)
    }

    alpha = 0.1 # 90% coverage

    for name, model in base_models.items():
        print(f"Running Conformal Prediction (MAPIE) for {name}...")

        # Standard Split Conformal or CV+
        # Using CV+ (method='plus') as it's robust
        try:
            mapie = MapieRegressor(estimator=model, method="plus", cv=5)
            mapie.fit(X_train, y_train)
            y_pred, y_pis = mapie.predict(X_test, alpha=alpha)
        except Exception as e:
            print(f"Error running MAPIE for {name}: {e}")
            continue

        # y_pis shape is (n_samples, 2, n_alpha)
        lower = y_pis[:, 0, 0]
        upper = y_pis[:, 1, 0]

        # Metrics
        coverage = regression_coverage_score(y_test, lower, upper)
        width = np.mean(upper - lower)

        metrics = {'Coverage': coverage, 'Mean Width': width}
        save_values_to_excel(metrics, excel_path, f"{name}_Metrics")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        # Sort
        sorted_idx = np.argsort(y_test)

        ax.errorbar(np.arange(len(y_test)), y_pred[sorted_idx],
                    yerr=[y_pred[sorted_idx] - lower[sorted_idx], upper[sorted_idx] - y_pred[sorted_idx]],
                    fmt='o', alpha=0.5, label='Prediction Interval')
        ax.plot(np.arange(len(y_test)), y_test[sorted_idx], 'r.', label='Actual')
        ax.set_title(f"{name} Conformal Prediction (alpha={alpha})")
        ax.legend()

        save_plot_to_excel(fig, excel_path, f"{name}_Plot")
        plt.close(fig)

    # NEXCP / Adaptive CP logic
    # Implementing a basic adaptive CP (CQR or similar) logic if not provided by library
    # The notebooks mentioned "NEXCP".
    # For now, we stick to standard MAPIE functionality which is robust.
    # If NEXCP (Non-exchangeable) is needed, it typically involves weighting.
    # MAPIE supports 'ensemble' and time-series splitting which approximates this.

    return "Conformal prediction analysis complete."
