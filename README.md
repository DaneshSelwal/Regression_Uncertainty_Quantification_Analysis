# 🚀 uq_regression: Comprehensive Regression Analysis Pipeline

`uq_regression` is a Python package designed for robust regression analysis, featuring state-of-the-art methods for **Uncertainty Quantification (UQ)**. It converts a complex folder-based workflow into a streamlined, installable library.

## 📦 Installation

Clone the repository and install the package:

```bash
git clone <repo_url>
cd <repo_directory>
pip install .
```

Or install dependencies manually:

```bash
pip install -r requirements.txt
```

## 🛠️ Modules Overview

The package is organized into the following modules:

*   **`uq_regression.hyperparameter_tuning`**:  Automated Bayesian optimization using Optuna.
*   **`uq_regression.quantile_regression`**: Interval estimation using Quantile Regression.
*   **`uq_regression.probabilistic_distribution`**: Full distribution modeling using NGBoost and PGBM.
*   **`uq_regression.card`**: Advanced probabilistic analysis (proxying CARD methodology).
*   **`uq_regression.conformal_prediction`**: Robust uncertainty quantification using Conformal Prediction (MAPIE).

## 🚀 Usage

You can import and run specific phases of the analysis in your own scripts.

### 1. Load Data

```python
from uq_regression.utils import load_data

X_train, y_train, X_test, y_test = load_data('Data_folder/Data/train.csv', 'Data_folder/Data/test.csv')
```

### 2. Hyperparameter Tuning

```python
from uq_regression.hyperparameter_tuning import hyperparameter_tuning_all, get_best_models_and_predict

# Output directory for results
output_dir = "results/tuning"

# Run optimization
best_scores = hyperparameter_tuning_all(X_train, y_train, X_test, y_test, output_dir)

# Get predictions from best models
predictions, models = get_best_models_and_predict(best_scores, X_train, y_train, X_test, y_test, output_dir)
```

### 3. Quantile Regression

```python
from uq_regression.quantile_regression import quantile_regression_analysis

output_dir = "results/quantile"
quantile_regression_analysis(X_train, y_train, X_test, y_test, output_dir)
```

### 4. Probabilistic Distribution (NGBoost, PGBM)

```python
from uq_regression.probabilistic_distribution import probabilistic_distribution_analysis

output_dir = "results/probabilistic"
probabilistic_distribution_analysis(X_train, y_train, X_test, y_test, output_dir)
```

### 5. Conformal Prediction

```python
from uq_regression.conformal_prediction import conformal_prediction_analysis

output_dir = "results/conformal"
conformal_prediction_analysis(X_train, y_train, X_test, y_test, output_dir)
```

## 📂 Output

All functions accept an `output_dir` argument. The package automatically saves:
*   **Excel Files**: Containing predictions, metrics, and inserted plots.
*   **Models**: Optimized models (in pickle format) where applicable.

---
*Based on the original repository work.*
