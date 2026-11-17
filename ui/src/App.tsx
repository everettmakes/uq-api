import { useEffect, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  ErrorBar,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// --- Types: mirror the API's JSON (like Pydantic models, but for the frontend) ---
type CurveResponse = {
  grid: number[];
  mean: number[];
  std: number[];
  train_x: number[];
  train_y: number[];
};

type PredictResponse = {
  run_id: string;
  mean: number[];
  std: number[];
};

// --- Types: the shapes the chart needs ---
type CurvePoint = { x: number; mean: number; band: [number, number] };
type Point = { x: number; y: number };
type Prediction = { x: number; mean: number; std: number; runId: string };

function App() {
  const [curve, setCurve] = useState<CurvePoint[]>([]);
  const [training, setTraining] = useState<Point[]>([]);
  const [xInput, setXInput] = useState("2.5");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load the curve once, when the page first renders
  useEffect(() => {
    fetch(`${API_URL}/curve`)
      .then((res) => {
        if (!res.ok) throw new Error(`API returned ${res.status}`);
        return res.json() as Promise<CurveResponse>;
      })
      .then((data) => {
        setCurve(
          data.grid.map((x, i): CurvePoint => ({
            x,
            mean: data.mean[i],
            band: [data.mean[i] - 2 * data.std[i], data.mean[i] + 2 * data.std[i]],
          })),
        );
        setTraining(data.train_x.map((x, i): Point => ({ x, y: data.train_y[i] })));
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  async function handlePredict() {
    const x = Number(xInput);
    if (xInput.trim() === "" || Number.isNaN(x)) {
      setError("Enter a number for x");
      return;
    }
    setError(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x: [[x]] }),
      });
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const data: PredictResponse = await res.json();
      setPrediction({ x, mean: data.mean[0], std: data.std[0], runId: data.run_id });
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const predictionPoint = prediction
    ? [{ x: prediction.x, y: prediction.mean, err: 2 * prediction.std }]
    : [];

  return (
    <div className="app">
      <h1>uq-api</h1>
      <p className="subtitle">Gaussian process predictions with uncertainty</p>

      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart data={curve} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="x"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v: number) => v.toFixed(1)}
          />
          <YAxis tickFormatter={(v: number) => v.toFixed(1)} />
          <Legend />
          <Area
            dataKey="band"
            name="±2 std"
            stroke="none"
            fill="#8884d8"
            fillOpacity={0.25}
            isAnimationActive={false}
          />
          <Line
            dataKey="mean"
            name="GP mean"
            stroke="#5a4fcf"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Scatter data={training} dataKey="y" name="Training data" fill="#222" />
          {prediction && (
            <ReferenceLine x={prediction.x} stroke="#e4572e" strokeDasharray="4 4" />
          )}
          {prediction && (
            <Scatter data={predictionPoint} dataKey="y" name="Your prediction" fill="#e4572e">
              <ErrorBar dataKey="err" direction="y" width={8} stroke="#e4572e" />
            </Scatter>
          )}
        </ComposedChart>
      </ResponsiveContainer>

      <div className="controls">
        <label htmlFor="x-input">x =</label>
        <input
          id="x-input"
          type="number"
          step="0.1"
          value={xInput}
          onChange={(e) => setXInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handlePredict()}
        />
        <button onClick={handlePredict}>Predict</button>
      </div>

      {error && <p className="error">{error}</p>}

      {prediction && (
        <div className="result">
          <dl>
            <dt>mean</dt>
            <dd>{prediction.mean.toFixed(3)}</dd>
            <dt>std</dt>
            <dd>{prediction.std.toFixed(3)}</dd>
            <dt>95% interval</dt>
            <dd>
              [{(prediction.mean - 2 * prediction.std).toFixed(3)},{" "}
              {(prediction.mean + 2 * prediction.std).toFixed(3)}]
            </dd>
            <dt>run id</dt>
            <dd>{prediction.runId}</dd>
          </dl>
        </div>
      )}
    </div>
  );
}

export default App;