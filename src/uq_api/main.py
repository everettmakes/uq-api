from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from uq_api.model import GPSurrogate
from uq_api.repository import InMemoryRunRepository
from uq_api.service import PredictionService


# --- Pydantic: the shape of the data coming in and going out ---
class PredictRequest(BaseModel):
    x: list[list[float]] = Field(min_length=1)


class PredictResponse(BaseModel):
    run_id: str
    mean: list[float]
    std: list[float]


# --- Build the service once, when the app starts ---
def build_service() -> PredictionService:
    X = np.array([0, 0.5, 1, 1.5, 3.5, 4, 4.5, 5]).reshape(-1, 1)
    rng = np.random.default_rng(seed=0)
    y = np.sin(X).ravel() + rng.normal(0, 0.1, size=len(X))
    model = GPSurrogate()
    model.fit(X, y)
    return PredictionService(model=model, repo=InMemoryRunRepository())


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = build_service()   # runs at startup
    yield                                 # app serves requests here


app = FastAPI(title="uq-api", lifespan=lifespan)


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