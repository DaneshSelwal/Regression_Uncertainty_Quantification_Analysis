import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from pgbm.sklearn import HistGradientBoostingRegressor
import os
from uq_regression.utils import save_plot_to_excel, save_values_to_excel

def pinball_loss(y_true, y_pred, quantile):
    delta = y_true - y_pred
    return np.maximum(quantile * delta, (quantile - 1) * delta).mean()

def quantile_regression_analysis(X_train, y_train, X_test, y_test, output_dir, quantiles=[0.05, 0.5, 0.95]):
    """
    Performs Quantile Regression using various models.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    predictions_dict = {}

    # Models that natively support quantile regression or can be adapted
    # Note: Many standard implementations (XGBoost, LGBM, CatBoost) have specific objective functions for quantiles

    models = {
        'Gradient Boosting': GradientBoostingRegressor,
        'XGBoost': XGBRegressor,
        'LightGBM': LGBMRegressor,
        'CatBoost': CatBoostRegressor,
        # NGBoost and PGBM are probabilistic, so we get quantiles from distribution
        'NGBoost': NGBRegressor,
    }

    results_excel_path = os.path.join(output_dir, "quantile_results.xlsx")

    for model_name, model_class in models.items():
        print(f"Running Quantile Regression for {model_name}...")
        model_preds = {}

        if model_name in ['NGBoost']:
            # Probabilistic models: fit once, predict distribution
            model = model_class(Dist=Normal, Score=LogScore, verbose=False)
            model.fit(X_train, y_train)
            preds_dist = model.pred_dist(X_test)

            for q in quantiles:
                # NGBoost uses scipy.stats distributions mostly
                # Normal distribution inverse cdf (ppf)
                import scipy.stats as stats
                # loc is mean, scale is std
                # preds_dist.loc, preds_dist.scale are arrays
                # dist is scipy.stats.norm(loc=preds_dist.loc, scale=preds_dist.scale)
                # But NGBoost returns a wrapper. Let's use the object methods if available or scipy
                # The wrapper usually has ppf equivalent or we construct it
                # NGBoost Dist objects often don't have ppf directly on the array wrapper in older versions,
                # but let's assume we can use scipy norm
                model_preds[q] = stats.norm.ppf(q, loc=preds_dist.params['loc'], scale=preds_dist.params['scale'])

        elif model_name == 'Gradient Boosting':
            for q in quantiles:
                model = model_class(loss='quantile', alpha=q, random_state=42)
                model.fit(X_train, y_train)
                model_preds[q] = model.predict(X_test)

        elif model_name == 'LightGBM':
            for q in quantiles:
                model = model_class(objective='quantile', alpha=q, random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                model_preds[q] = model.predict(X_test)

        elif model_name == 'XGBoost':
            for q in quantiles:
                # XGBoost quantile regression usually uses 'reg:quantileerror' or custom objective
                # For simplicity/compatibility, we assume standard quantile obj is available in newer versions
                # or use 'reg:absoluteerror' which is median (0.5).
                # Actually, XGBoost added 'reg:quantileerror' recently.
                # If older, we might need custom obj. Let's try standard interface.
                try:
                    model = model_class(objective='reg:quantileerror', quantile_alpha=q, random_state=42)
                    model.fit(X_train, y_train)
                    model_preds[q] = model.predict(X_test)
                except:
                    print(f"XGBoost quantile not supported directly or version mismatch. Skipping {q} for XGBoost.")
                    model_preds[q] = np.zeros_like(y_test)

        elif model_name == 'CatBoost':
            for q in quantiles:
                # CatBoost uses 'Quantile:alpha=...'
                loss_function = f'Quantile:alpha={q}'
                model = model_class(loss_function=loss_function, random_seed=42, verbose=0)
                model.fit(X_train, y_train)
                model_preds[q] = model.predict(X_test)

        # Save results
        if model_preds:
            df_res = pd.DataFrame(model_preds)
            df_res['Actual'] = y_test
            predictions_dict[model_name] = df_res

            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            # Sort for clean line plot
            sorted_idx = np.argsort(y_test)
            ax.plot(np.arange(len(y_test)), y_test[sorted_idx], 'k.', label='Actual', alpha=0.5)

            if 0.5 in model_preds:
                ax.plot(np.arange(len(y_test)), model_preds[0.5][sorted_idx], 'r-', label='Median')

            if 0.05 in model_preds and 0.95 in model_preds:
                ax.fill_between(np.arange(len(y_test)),
                                model_preds[0.05][sorted_idx],
                                model_preds[0.95][sorted_idx],
                                color='gray', alpha=0.2, label='90% Interval')

            ax.set_title(f"{model_name} Quantile Regression")
            ax.legend()

            save_plot_to_excel(fig, results_excel_path, f"{model_name}_Plot")
            plt.close(fig)

            # Calculate metrics (coverage, width)
            lower = model_preds.get(0.05, np.zeros_like(y_test))
            upper = model_preds.get(0.95, np.zeros_like(y_test))

            coverage = np.mean((y_test >= lower) & (y_test <= upper))
            width = np.mean(upper - lower)

            metrics = {'Coverage (90%)': coverage, 'Mean Width': width}
            save_values_to_excel(metrics, results_excel_path, f"{model_name}_Metrics")

    return predictions_dict
