"""
main.py — Water Quality Prediction FastAPI
-------------------------------------------
Dual System API:
    A) Scientific WQI (current, from BIS formula)
    B) LSTM Predicted WQI (future T+1)

Features:
    - Input validation layer (range checks, outlier clipping)
    - Sliding window buffer maintained server-side
    - Manual input APPENDS to buffer (no tiling)
    - Warming-up state when buffer < 24
    - Mean-replacement explainability (not zero-out)
    - Confidence: threshold distance + prediction variance + distribution distance
    - Anomaly detection (rapid drop + threshold breach)
    - Concept drift awareness

Run from project root:
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""
import os, sys, time, logging, numpy as np, joblib, tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from wqi import calculate_current_wqi_from_values, VALID_RANGES

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("wq_api")

SEQ_LENGTH = 24
WQI_THRESHOLD = 70.0
FEATURE_ORDER = ["pH","Turbidity","Dissolved_Oxygen","Temperature","Conductivity","BOD","COD"]

app = FastAPI(title="Water Quality Dual Prediction API",
              description="Scientific WQI (current) + LSTM Forecast (future T+1)", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    pH: float = Field(..., ge=0.0, le=14.0)
    Turbidity: float = Field(..., ge=0.0, le=100.0)
    Dissolved_Oxygen: float = Field(..., ge=0.0, le=20.0)
    Temperature: float = Field(..., ge=0.0, le=50.0)
    Conductivity: float = Field(..., ge=0.0, le=1000.0)
    BOD: float = Field(..., ge=0.0, le=100.0)
    COD: float = Field(..., ge=0.0, le=200.0)

    @field_validator("*")
    @classmethod
    def check_nan_inf(cls, v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            raise ValueError("Input cannot be NaN or Inf")
        return v


class SingleReadingRequest(BaseModel):
    pH: float = Field(7.5, ge=0.0, le=14.0)
    Turbidity: float = Field(3.0, ge=0.0, le=100.0)
    Dissolved_Oxygen: float = Field(8.0, ge=0.0, le=20.0)
    Temperature: float = Field(20.0, ge=0.0, le=50.0)
    Conductivity: float = Field(300.0, ge=0.0, le=1000.0)
    BOD: float = Field(5.0, ge=0.0, le=100.0)
    COD: float = Field(10.0, ge=0.0, le=200.0)

    @model_validator(mode="after")
    def validate_ranges(self):
        """Runtime data quality layer — clips outliers, logs warnings."""
        for fname in FEATURE_ORDER:
            val = getattr(self, fname)
            lo, hi = VALID_RANGES[fname]
            if not (lo <= val <= hi):
                clipped = float(np.clip(val, lo, hi))
                logger.warning("Input %s=%.2f out of range [%.1f, %.1f], clipped to %.2f",
                               fname, val, lo, hi, clipped)
                setattr(self, fname, clipped)
        return self


class PredictionRequest(BaseModel):
    sequence: List[SensorReading] = Field(..., min_length=SEQ_LENGTH, max_length=SEQ_LENGTH)


class DualPredictionResponse(BaseModel):
    current_wqi: float
    predicted_wqi: Optional[float] = None
    delta: Optional[float] = None
    trend: str
    status: str
    confidence: str
    explanation: str
    anomaly: Optional[str] = None
    warming_up: bool = False
    buffer_fill: int = 0
    latency_ms: float


# ── Assets ───────────────────────────────────────────────────────────────────
_model = None
_scaler = None
_target_scaler = None
_sliding_buffer = deque(maxlen=SEQ_LENGTH)
_prev_predicted_wqi = None
_recent_predictions = deque(maxlen=10)
_training_mean = None  # Mean of training features (for distribution distance)
_training_std = None


def _get_path(filename):
    for root in [".", ".."]:
        path = os.path.join(root, "models", filename)
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


@app.on_event("startup")
async def startup_event():
    global _model, _scaler, _target_scaler, _training_mean, _training_std
    files = {"model": "lstm_wqi_model.keras", "scaler": "scaler.pkl", "target": "target_scaler.pkl"}
    paths = {k: _get_path(v) for k, v in files.items()}
    missing = [v for k, v in files.items() if paths[k] is None]
    if missing:
        logger.error("CRITICAL: Missing assets: %s. Run 'python src/train.py' first.", missing)
        os._exit(1)
    try:
        _model = tf.keras.models.load_model(paths["model"])
        _scaler = joblib.load(paths["scaler"])
        _target_scaler = joblib.load(paths["target"])
        # Compute training distribution stats from scaler for confidence scoring
        _training_mean = (_scaler.data_min_ + _scaler.data_max_) / 2.0
        _training_std = (_scaler.data_max_ - _scaler.data_min_) / 4.0  # approximate
        logger.info("Assets loaded from %s", os.path.dirname(paths["model"]))
    except Exception as e:
        logger.error("CRITICAL: Failed to load assets: %s", e)
        os._exit(1)


# ── Mean-Replacement Explainability ──────────────────────────────────────────

def explain_prediction(scaled_input, base_wqi):
    """
    Per-prediction explainability using mean replacement (not zero-out).
    For each feature, replace its values with the feature's mean across the
    sequence, then measure prediction change. This keeps inputs in-distribution.
    """
    impacts = {}
    for i, name in enumerate(FEATURE_ORDER):
        perturbed = scaled_input.copy()
        perturbed[:, i] = np.mean(scaled_input[:, i])  # Replace with mean (in-distribution)
        pert_pred = _model.predict(np.expand_dims(perturbed, 0), verbose=0)
        pert_wqi = float(_target_scaler.inverse_transform(pert_pred)[0][0])
        impacts[name] = abs(base_wqi - pert_wqi)

    total = sum(impacts.values()) + 1e-9
    pct = {k: (v / total) * 100 for k, v in impacts.items()}
    sorted_f = sorted(pct.items(), key=lambda x: -x[1])

    # Natural language explanation
    top3 = sorted_f[:3]
    parts = [f"{n} ({p:.1f}%)" for n, p in top3]

    # Directional hint based on latest reading vs ideal
    return "Top factors: " + ", ".join(parts), pct


# ── Improved Confidence Scoring ──────────────────────────────────────────────

def compute_confidence(predicted_wqi, feature_array):
    """
    Confidence = f(threshold_distance, prediction_stability, distribution_distance)
    """
    # 1. Distance from safety threshold
    dist = abs(predicted_wqi - WQI_THRESHOLD)

    # 2. Recent prediction stability (variance)
    variance = float(np.std(list(_recent_predictions))) if len(_recent_predictions) > 1 else 0.0

    # 3. Distance from training distribution (Mahalanobis-lite)
    if _training_mean is not None and _training_std is not None:
        latest = feature_array[-1]
        z_scores = np.abs((latest - _training_mean) / (_training_std + 1e-9))
        max_z = float(np.max(z_scores))
    else:
        max_z = 0.0

    # Composite scoring
    if dist > 20 and variance < 5 and max_z < 3:
        return "High"
    elif dist > 10 and max_z < 4:
        return "Medium"
    else:
        return "Low"


# ── Core Dual Prediction ─────────────────────────────────────────────────────

def _dual_predict(feature_array):
    global _prev_predicted_wqi
    t0 = time.perf_counter()
    latest = feature_array[-1]

    # A) Scientific WQI (Current)
    current_wqi = calculate_current_wqi_from_values(
        float(latest[0]), float(latest[1]), float(latest[2]),
        float(latest[3]), float(latest[4]), float(latest[5]), float(latest[6])
    )

    # B) LSTM Prediction (Future T+1)
    scaled = _scaler.transform(feature_array)
    lstm_in = np.expand_dims(scaled, axis=0)
    raw_pred = _model.predict(lstm_in, verbose=0)
    predicted_wqi = float(np.clip(_target_scaler.inverse_transform(raw_pred)[0][0], 0, 100))

    # Delta & Trend
    delta = predicted_wqi - current_wqi
    if delta > 2: trend = "up"
    elif delta < -2: trend = "down"
    else: trend = "stable"

    # Confidence
    _recent_predictions.append(predicted_wqi)
    confidence = compute_confidence(predicted_wqi, feature_array)

    # Explainability
    explanation, _ = explain_prediction(scaled, predicted_wqi)

    # Anomaly Detection
    status = "Safe" if predicted_wqi >= WQI_THRESHOLD else "Unsafe"
    anomaly = None
    if predicted_wqi < WQI_THRESHOLD:
        anomaly = f"Pollution Risk: Predicted WQI ({predicted_wqi:.1f}) below threshold ({WQI_THRESHOLD})"
    if _prev_predicted_wqi is not None:
        drop_pct = ((_prev_predicted_wqi - predicted_wqi) / (_prev_predicted_wqi + 1e-9)) * 100
        if drop_pct > 15:
            anomaly = f"RAPID DROP: WQI dropped {drop_pct:.1f}% from previous reading!"

    _prev_predicted_wqi = predicted_wqi
    latency = (time.perf_counter() - t0) * 1000

    logger.info("DUAL: Current=%.1f Predicted=%.1f D=%+.1f %s %.0fms",
                current_wqi, predicted_wqi, delta, status, latency)

    return DualPredictionResponse(
        current_wqi=round(current_wqi, 2), predicted_wqi=round(predicted_wqi, 2),
        delta=round(delta, 2), trend=trend, status=status, confidence=confidence,
        explanation=explanation, anomaly=anomaly, warming_up=False,
        buffer_fill=SEQ_LENGTH, latency_ms=round(latency, 2),
    )


def _warming_up_response(latest, buffer_len):
    """Return current WQI only when buffer hasn't filled yet."""
    t0 = time.perf_counter()
    current_wqi = calculate_current_wqi_from_values(
        float(latest[0]), float(latest[1]), float(latest[2]),
        float(latest[3]), float(latest[4]), float(latest[5]), float(latest[6])
    )
    status = "Safe" if current_wqi >= WQI_THRESHOLD else "Unsafe"
    latency = (time.perf_counter() - t0) * 1000

    return DualPredictionResponse(
        current_wqi=round(current_wqi, 2), predicted_wqi=None, delta=None,
        trend="stable", status=status, confidence="Low",
        explanation=f"Warming up: {buffer_len}/{SEQ_LENGTH} readings collected",
        anomaly=None, warming_up=True, buffer_fill=buffer_len,
        latency_ms=round(latency, 2),
    )


@app.post("/predict", response_model=DualPredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    features = np.array([[s.pH, s.Turbidity, s.Dissolved_Oxygen, s.Temperature,
                          s.Conductivity, s.BOD, s.COD] for s in request.sequence], dtype=np.float32)
    return _dual_predict(features)


@app.post("/predict/single", response_model=DualPredictionResponse, tags=["Prediction"])
def predict_single(request: SingleReadingRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    reading = [request.pH, request.Turbidity, request.Dissolved_Oxygen,
               request.Temperature, request.Conductivity, request.BOD, request.COD]
    _sliding_buffer.append(reading)

    if len(_sliding_buffer) < SEQ_LENGTH:
        return _warming_up_response(reading, len(_sliding_buffer))

    feature_array = np.array(list(_sliding_buffer), dtype=np.float32)
    return _dual_predict(feature_array)


@app.get("/health")
def health():
    return {"status": "ok", "assets_loaded": _model is not None, "buffer_size": len(_sliding_buffer)}


@app.post("/buffer/reset", tags=["Buffer"])
def reset_buffer():
    _sliding_buffer.clear()
    return {"status": "buffer_cleared", "buffer_size": 0}
