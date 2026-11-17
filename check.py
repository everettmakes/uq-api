import numpy as np

from uq_api.model import GPSurrogate
from uq_api.repository import InMemoryRunRepository
from uq_api.service import PredictionService

X = np.array([0, 0.5, 1, 1.5, 3.5, 4, 4.5, 5]).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, size=len(X))

model = GPSurrogate()
model.fit(X, y)

repo = InMemoryRunRepository()
service = PredictionService(model=model, repo=repo)

result = service.predict(np.array([[0.5], [2.5], [6.0]]))
print(result)
print(repo.get(result["run_id"]))