# uq-api
![CI](https://github.com/everettmakes/uq-api/actions/workflows/ci.yml/badge.svg)

A containerised REST API that returns predictions **with uncertainty**, using a Gaussian process surrogate model.

![uq-api UI showing the GP mean, uncertainty band and a prediction](docs/screenshot.png)

Every prediction comes back as a mean and a standard deviation, and every run is stored in MongoDB so it can be retrieved later.

```json
POST /predict  {"x": [[0.5], [2.5], [6.0]]}

{
  "run_id": "f0336424-...",
  "mean": [0.488, 0.493, -0.635],
  "std":  [0.036, 0.066, 0.222]
}
```

The standard deviation grows as inputs move away from the training data. At `x = 6.0`, beyond the data, the prediction is least accurate, and the model reports that by returning its largest uncertainty.

---

## Why uncertainty?

Surrogate models stand in for expensive simulations. A fast approximation is only useful if you know where to trust it. A Gaussian process gives you both: a prediction, and a measure of how far that prediction is from anything the model has seen.

---

## Architecture

```mermaid
flowchart LR
    C[Client] -->|HTTP| A[FastAPI<br/>main.py]
    A -->|validated input| S[PredictionService<br/>service.py]
    S --> M[Surrogate<br/>Protocol]
    S --> R[RunRepository<br/>Protocol]
    M -.implemented by.-> GP[GPSurrogate<br/>model.py]
    R -.implemented by.-> MR[MongoRunRepository]
    R -.implemented by.-> IR[InMemoryRunRepository]
    MR --> DB[(MongoDB)]
```

| Layer | File | Responsibility |
|---|---|---|
| HTTP | `main.py` | Routing, Pydantic validation, status codes |
| Service | `service.py` | Calls the model, records the run |
| Model | `model.py` | Gaussian process (scikit-learn, RBF + white noise kernel) |
| Storage | `repository.py` | Saves and retrieves runs (MongoDB or in-memory) |

### Design decisions

- **Dependency inversion.** `PredictionService` depends on two `Protocol` interfaces, `Surrogate` and `RunRepository`, never on concrete classes. Swapping scikit-learn for GPyTorch, or MongoDB for another store, needs no change to the service.
- **Repository pattern.** Storage sits behind `save` and `get`. Production uses MongoDB; tests and quick local runs use an in-memory store.
- **Configuration from the environment.** If `MONGO_URL` is set, runs go to MongoDB. If not, they stay in memory.
- **Train once at startup.** The model is fitted in FastAPI's `lifespan` handler, not per request. The database connection is opened and closed there too.
- **Validation at the boundary.** Pydantic rejects malformed input with a `422` before any application code runs.

---

## Quickstart

Requires [Docker](https://www.docker.com/).

```bash
docker compose up --build
```

Then open **http://localhost:8000/docs** for the interactive API documentation.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | Predict mean and std for one or more inputs |
| `GET` | `/runs/{run_id}` | Retrieve a stored run (`404` if not found) |

Example with curl:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"x": [[0.5], [2.5]]}'
```

Stop with `docker compose down`. Add `-v` to delete stored runs as well.

---

## Local development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run uvicorn uq_api.main:app --reload
```

Without `MONGO_URL`, the app uses the in-memory repository, so no database is needed.

---

## Testing

The suite follows the testing pyramid. Each level isolates a different layer.

```
        E2E          5   full HTTP round trips via TestClient
     Integration     2   MongoRunRepository against a real database
      Component      3   PredictionService with fake model and repository
        Unit         4   GPSurrogate on its own
```

| File | Level | What it checks |
|---|---|---|
| `test_model.py` | Unit | Output shapes, non-negative std, **std grows away from data**, fit to training points |
| `test_service.py` | Component | Service passes model output through, saves every run, returns the repository's ID |
| `test_mongo_repository.py` | Integration | Save/get round trip and missing IDs against real MongoDB |
| `test_api.py` | End-to-end | Health check, prediction, storage round trip, `404` and `422` handling |

Component tests use fakes that satisfy the `Protocol` interfaces, so a failure points to the service itself rather than to scikit-learn or the database.

**Run without a database** (the MongoDB tests are skipped):

```bash
uv run pytest -v
```

**Run everything, including MongoDB:**

```bash
docker compose up -d mongo
MONGO_URL=mongodb://localhost:27017 uv run pytest -v
```

---

## Project structure

```
uq-api/
├── src/uq_api/
│   ├── main.py          FastAPI app, endpoints, startup
│   ├── service.py       PredictionService
│   ├── model.py         Surrogate protocol, GPSurrogate
│   └── repository.py    RunRepository protocol, Mongo and in-memory stores
├── tests/
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_service.py
│   ├── test_mongo_repository.py
│   └── test_api.py
├── Dockerfile
├── compose.yaml
└── pyproject.toml
```

---

## Limitations and next steps

- **Toy training data.** The model is trained at startup on noisy samples of `sin(x)`. A real deployment would load data or a trained model from storage.
- **Scaling.** Exact GP training costs O(n³) in the number of data points. Beyond a few thousand points, a sparse or approximate GP would be needed.
- **Planned:** CI with GitHub Actions, performance and load tests, and an endpoint to retrain the model on new data.

---

## Tech stack

Python 3.13 · FastAPI · Pydantic · scikit-learn · NumPy · MongoDB (PyMongo) · pytest · Docker · Docker Compose · uv
