import numpy as np

from uq_api.model import Surrogate
from uq_api.repository import RunRepository


class PredictionService:
    def __init__(self, model: Surrogate, repo: RunRepository) -> None:
        self.model = model
        self.repo = repo

    def predict(self, X: np.ndarray) -> dict:
        mean, std = self.model.predict(X)
        run_id = self.repo.save(X, mean, std)
        return {
            "run_id": run_id,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }