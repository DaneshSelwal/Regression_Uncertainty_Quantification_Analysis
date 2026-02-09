import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from uq_regression.probabilistic_distribution import probabilistic_distribution_analysis, mseloss_objective, rmseloss_metric
from uq_regression.utils import save_plot_to_excel, save_values_to_excel
from pgbm.torch import PGBM
import torch

def card_analysis(X_train, y_train, X_test, y_test, output_dir):
    """
    Placeholder for CARD (Classification and Regression Diffusion) analysis.

    Currently, based on the provided notebook 'Probabilistic__Distribution(CARD).ipynb',
    the analysis primarily utilizes probabilistic models like PGBM, NGBoost, etc.
    If a specific Diffusion-based CARD implementation is provided, it should be integrated here.

    For now, this function re-uses the probabilistic distribution analysis but saves to the CARD directory
    to match the user's workflow structure.
    """
    print("Running CARD Analysis (using Probabilistic Distribution models as proxy based on current notebook content)...")

    # We can reuse the probabilistic analysis logic, or implement the specific comparisons found in that notebook.
    # The notebook output showed PGBM calibration curves.

    # Let's run PGBM as a representative 'advanced' probabilistic model for this section
    # If the user provides the specific 'CARD' class (e.g. from a paper implementation), import it here.

    # Creating a subdirectory for CARD results
    card_output_dir = os.path.join(output_dir, "CARD_Results")
    os.makedirs(card_output_dir, exist_ok=True)

    # Re-using the PGBM logic from probabilistic module for consistency
    # In a real scenario, this would import the diffusion model class.

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print("Training PGBM (CARD proxy)...")
    pgbm = PGBM()
    # Using typical params found in notebooks
    pgbm.train((X_train, y_train), objective=mseloss_objective, metric=rmseloss_metric, params={'n_estimators': 500, 'learning_rate': 0.01, 'verbose': 0})

    pgbm_dist = pgbm.predict_dist(X_test)

    if isinstance(pgbm_dist, tuple):
        pgbm_mu = pgbm_dist[0]
        pgbm_sigma = pgbm_dist[1]
    elif torch.is_tensor(pgbm_dist):
        if pgbm_dist.ndim == 2:
            if pgbm_dist.shape[1] == len(X_test):
                pgbm_mu = pgbm_dist.mean(dim=0)
                pgbm_sigma = pgbm_dist.std(dim=0)
            elif pgbm_dist.shape[1] == 2 and pgbm_dist.shape[0] == len(X_test):
                pgbm_mu = pgbm_dist[:, 0]
                pgbm_sigma = pgbm_dist[:, 1]
            else:
                pgbm_mu = pgbm_dist
                pgbm_sigma = torch.ones_like(pgbm_dist)
        else:
            pgbm_mu = pgbm_dist
            pgbm_sigma = torch.ones_like(pgbm_dist)
    else:
        pgbm_mu = pgbm_dist
        pgbm_sigma = np.ones_like(pgbm_dist)

    # Ensure numpy
    if torch.is_tensor(pgbm_mu):
        pgbm_mu = pgbm_mu.detach().cpu().numpy()
    if torch.is_tensor(pgbm_sigma):
        pgbm_sigma = pgbm_sigma.detach().cpu().numpy()

    # Save results
    excel_path = os.path.join(card_output_dir, "card_proxy_results.xlsx")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_idx = np.argsort(y_test)
    ax.errorbar(np.arange(len(y_test)), pgbm_mu[sorted_idx], yerr=2*pgbm_sigma[sorted_idx], fmt='o', alpha=0.5, label='Predicted (mean ± 2σ)')
    ax.plot(np.arange(len(y_test)), y_test[sorted_idx], 'r.', label='Actual')
    ax.set_title("PGBM (CARD Phase) Probabilistic Prediction")
    ax.legend()

    save_plot_to_excel(fig, excel_path, "PGBM_Plot")
    plt.close(fig)

    print(f"CARD analysis (proxy) complete. Results saved to {card_output_dir}")

    return {"PGBM": (pgbm_mu, pgbm_sigma)}
