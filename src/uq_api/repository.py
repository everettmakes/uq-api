import uuid
from typing import Protocol

import numpy as np
from pymongo.collection import Collection


class RunRepository(Protocol):
    def save(self, X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> str: ...
    def get(self, run_id: str) -> dict | None: ...


class InMemoryRunRepository:
    def __init__(self) -> None:
        self._runs: dict[str, dict] = {}

    def save(self, X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> str:
        run_id = str(uuid.uuid4())
        self._runs[run_id] = {
            "x": X.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
        return run_id

    def get(self, run_id: str) -> dict | None:
        return self._runs.get(run_id)


class MongoRunRepository:
    def __init__(self, collection: Collection) -> None:
        self._runs = collection

    def save(self, X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> str:
        run_id = str(uuid.uuid4())
        self._runs.insert_one(
            {
                "_id": run_id,
                "x": X.tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
        )
        return run_id

    def get(self, run_id: str) -> dict | None:
        return self._runs.find_one({"_id": run_id}, {"_id": 0})
