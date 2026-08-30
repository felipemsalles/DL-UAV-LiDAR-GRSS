"""Crown allometry: ln V = a + sum_i b_i * ln(x_i). Primary volume model at N≈23.

Interpretable, robust at small N, directly comparable to the stand's own (G2/Schumacher) equations.
Operates on numpy arrays so it composes with greenvista.eval.validation.loocv_predict.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

class CrownAllometry:
    def __init__(self):
        self.lr = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, float); y = np.asarray(y, float)
        if np.any(X <= 0) or np.any(y <= 0):
            raise ValueError("CrownAllometry needs strictly positive features and target (log-log).")
        self.lr.fit(np.log(X), np.log(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.lr.predict(np.log(np.asarray(X, float))))
