import random

from locust import HttpUser, between, task


class UQUser(HttpUser):
    wait_time = between(0.1, 0.5)   # each simulated user pauses 0.1–0.5s between actions

    @task(5)
    def predict(self):
        x = random.uniform(-1, 6)
        self.client.post("/predict", json={"x": [[x]]})

    @task(1)
    def curve(self):
        self.client.get("/curve", params={"n": 200})

    @task(1)
    def health(self):
        self.client.get("/health")