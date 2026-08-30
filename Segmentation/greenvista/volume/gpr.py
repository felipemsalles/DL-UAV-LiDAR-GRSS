"""Gaussian Process regression for volume — gives prediction intervals (valued in forestry)."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

class CrownGPR:
    def __init__(self, random_state: int = 42):
        self.scaler = StandardScaler()
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=2, random_state=random_state)

    def fit(self, X, y):
        self.gp.fit(self.scaler.fit_transform(np.asarray(X, float)), np.asarray(y, float))
        return self

    def predict(self, X, return_std: bool = False):
        Xs = self.scaler.transform(np.asarray(X, float))
        return self.gp.predict(Xs, return_std=return_std)
