import numpy as np

from uq_api.service import PredictionService


class FakeModel:
    """Returns fixed values, so the test checks the service, not the GP."""
    def fit(self, X, y): pass
    def predict(self, X):
        n = len(X)
        return np.full(n, 1.0), np.full(n, 0.1)


class FakeRepo:
    """Records what was saved, so the test can inspect it."""
    def __init__(self):
        self.saved = []
    def save(self, X, mean, std):
        self.saved.append((X, mean, std))
        return "fake-id"
    def get(self, run_id):
        return None


def test_service_returns_model_output():
    service = PredictionService(model=FakeModel(), repo=FakeRepo())
    result = service.predict(np.array([[0.5], [1.0]]))
    assert result["mean"] == [1.0, 1.0]
    assert result["std"] == [0.1, 0.1]


def test_service_saves_every_prediction():
    repo = FakeRepo()
    service = PredictionService(model=FakeModel(), repo=repo)
    service.predict(np.array([[0.5]]))
    assert len(repo.saved) == 1


def test_service_returns_run_id_from_repo():
    service = PredictionService(model=FakeModel(), repo=FakeRepo())
    result = service.predict(np.array([[0.5]]))
    assert result["run_id"] == "fake-id"