"""Trivial baselines for the volume comparison ladder."""
import numpy as np

class MeanRegressor:
    """Predicts the training-set mean (the 'do nothing' baseline; rRMSE = target CV)."""
    def __init__(self):
        self.mean_ = None

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else np.asarray(X).shape[0]
        return np.full(n, self.mean_, dtype=float)
