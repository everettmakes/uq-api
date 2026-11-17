from typing import Protocol

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class Surrogate(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class GPSurrogate:
    def __init__(self) -> None:
        kernel = RBF() + WhiteKernel()
        self._gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._gp.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self._gp.predict(X, return_std=True)
        return mean, std