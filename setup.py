from setuptools import setup, find_packages

setup(
    name="uq_regression",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "ngboost",
        "pgbm",
        "gpboost",
        "mapie",
        "puncc",
        "pytorch-tabnet",
        "optuna",
        "optunahub",
        "openpyxl",
        "scipy",
        "torch",
        "pillow"
    ],
    description="Comprehensive Regression Analysis Pipeline with Uncertainty Quantification",
    author="Your Name",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
