import sklearn.utils.fixes
import numpy as np

# Monkey patch for PGBM compatibility with newer scikit-learn
if not hasattr(sklearn.utils.fixes, 'percentile'):
    sklearn.utils.fixes.percentile = np.percentile
