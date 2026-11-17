import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient

from uq_api.model import GPSurrogate
from uq_api.repository import InMemoryRunRepository, MongoRunRepository, RunRepository
from uq_api.service import PredictionService


# --- Pydantic: the shape of the data coming in and going out ---
class PredictRequest(BaseModel):
    x: list[list[float]] = Field(min_length=1)


class PredictResponse(BaseModel):
    run_id: str
    mean: list[float]
    std: list[float]


# --- Build the service once, when the app starts ---
def make_training_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([0, 0.5, 1, 1.5, 3.5, 4, 4.5, 5]).reshape(-1, 1)
    rng = np.random.default_rng(seed=0)
    y = np.sin(X).ravel() + rng.normal(0, 0.1, size=len(X))
    return X, y


def build_service(
    repo: RunRepository, X: np.ndarray, y: np.ndarray
) -> PredictionService:
    model = GPSurrogate()
    model.fit(X, y)
    return PredictionService(model=model, repo=repo)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo_url = os.getenv("MONGO_URL")
    client = None
    if mongo_url:
        client = MongoClient(mongo_url)
        repo = MongoRunRepository(client["uq"]["runs"])
    else:
        repo = InMemoryRunRepository()

    X_train, y_train = make_training_data()
    app.state.training = {"x": X_train.ravel().tolist(), "y": y_train.tolist()}
    app.state.service = build_service(repo, X_train, y_train)
    yield  # app serves requests here
    if client is not None:
        client.close()  # shutdown: close the DB connection


app = FastAPI(title="uq-api", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    X = np.array(request.x)
    return app.state.service.predict(X)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = app.state.service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/curve")
def curve(
    x_min: float = -1.0,
    x_max: float = 6.0,
    n: int = Query(200, ge=2, le=2000),
) -> dict:
    if x_min >= x_max:
        raise HTTPException(status_code=400, detail="x_min must be less than x_max")
    grid = np.linspace(x_min, x_max, n).reshape(-1, 1)
    mean, std = app.state.service.preview(grid)
    return {
        "grid": grid.ravel().tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "train_x": app.state.training["x"],
        "train_y": app.state.training["y"],
    }
