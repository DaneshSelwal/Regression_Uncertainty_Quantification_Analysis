import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from pgbm.torch import PGBM
import torch
from uq_regression.utils import save_plot_to_excel, save_values_to_excel

def mseloss_objective(yhat, y, sample_weight=None):
    if not torch.is_tensor(yhat):
        yhat = torch.from_numpy(np.array(yhat)).float()
    if not torch.is_tensor(y):
        y = torch.from_numpy(np.array(y)).float()
    gradient = yhat - y
    hessian = torch.ones_like(yhat)
    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    if not torch.is_tensor(yhat):
        yhat = torch.from_numpy(np.array(yhat)).float()
    if not torch.is_tensor(y):
        y = torch.from_numpy(np.array(y)).float()
    loss = torch.sqrt(torch.mean((yhat - y) ** 2))
    return loss

def probabilistic_distribution_analysis(X_train, y_train, X_test, y_test, output_dir):
    """
    Runs probabilistic regression models (NGBoost, PGBM).
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    excel_path = os.path.join(output_dir, "probabilistic_results.xlsx")

    # NGBoost
    print("Running NGBoost...")
    ngb = NGBRegressor(Dist=Normal, Score=LogScore, n_estimators=500, learning_rate=0.01, verbose=False)
    ngb.fit(X_train, y_train)
    ngb_preds_dist = ngb.pred_dist(X_test)
    ngb_mean = ngb.predict(X_test)
    # ngb_preds_dist.params['loc'] is mean, 'scale' is std for Normal
    ngb_std = ngb_preds_dist.params['scale']

    # PGBM
    print("Running PGBM...")
    pgbm = PGBM()
    pgbm.train((X_train, y_train), objective=mseloss_objective, metric=rmseloss_metric, params={'n_estimators': 500, 'learning_rate': 0.01, 'verbose': 0})
    pgbm_preds = pgbm.predict(X_test) # Point forecast
    pgbm_dist = pgbm.predict_dist(X_test) # Returns mean and variance? Or distribution object?

    # PGBM predict_dist usually returns mean and std (or variance) depending on version/config
    # checking repo/docs behavior: usually returns (mu, sigma)
    if isinstance(pgbm_dist, tuple):
        pgbm_mu = pgbm_dist[0]
        pgbm_sigma = pgbm_dist[1]
    elif torch.is_tensor(pgbm_dist):
        # If it returns a single tensor, check shape.
        if pgbm_dist.ndim == 2:
            # Case 1: (n_forecasts, n_samples) - Calculate mean/std from samples
            if pgbm_dist.shape[1] == len(X_test):
                pgbm_mu = pgbm_dist.mean(dim=0)
                pgbm_sigma = pgbm_dist.std(dim=0)
            # Case 2: (n_samples, 2) - Parameters returned directly
            elif pgbm_dist.shape[1] == 2 and pgbm_dist.shape[0] == len(X_test):
                pgbm_mu = pgbm_dist[:, 0]
                pgbm_sigma = pgbm_dist[:, 1]
            else:
                # Ambiguous case, fallback to simple mean
                pgbm_mu = pgbm_dist.mean(dim=0) if pgbm_dist.shape[1] == len(X_test) else pgbm_dist
                pgbm_sigma = torch.ones_like(pgbm_mu)
        else:
            pgbm_mu = pgbm_dist
            pgbm_sigma = torch.ones_like(pgbm_dist) # Dummy
    else:
        # Fallback if it returns just one thing or different structure
        print(f"PGBM predict_dist returned type: {type(pgbm_dist)}")
        pgbm_mu = pgbm_preds
        pgbm_sigma = np.ones_like(pgbm_preds) # Dummy

    # Ensure numpy
    if torch.is_tensor(pgbm_mu):
        pgbm_mu = pgbm_mu.detach().cpu().numpy()
    if torch.is_tensor(pgbm_sigma):
        pgbm_sigma = pgbm_sigma.detach().cpu().numpy()

    models = {
        'NGBoost': (ngb_mean, ngb_std),
        'PGBM': (pgbm_mu, pgbm_sigma)
    }

    for name, (mu, sigma) in models.items():
        # Plot calibration / PIT
        # Simplified: just plot Actual vs Predicted with error bars (2 std dev)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort for clearer plot
        sorted_idx = np.argsort(y_test)

        ax.errorbar(np.arange(len(y_test)), mu[sorted_idx], yerr=2*sigma[sorted_idx], fmt='o', alpha=0.5, label='Predicted (mean ± 2σ)')
        ax.plot(np.arange(len(y_test)), y_test[sorted_idx], 'r.', label='Actual')
        ax.set_title(f"{name} Probabilistic Prediction")
        ax.legend()

        save_plot_to_excel(fig, excel_path, f"{name}_Plot")
        plt.close(fig)

        # Metrics: NLL, CRPS (simplified)
        mse = mean_squared_error(y_test, mu)
        # NLL for Gaussian: 0.5 * log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2)
        nll = np.mean(0.5 * np.log(2 * np.pi * sigma**2) + (y_test - mu)**2 / (2 * sigma**2))

        metrics = {'MSE': mse, 'NLL': nll}
        save_values_to_excel(metrics, excel_path, f"{name}_Metrics")

    return models
