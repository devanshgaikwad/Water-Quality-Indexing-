"""
simulate.py — Real-Time Prediction Simulation for the Water Quality LSTM model.
Replays test portion of the dataset through the trained model using a sliding window.
"""
import os, sys, time, numpy as np, joblib, tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_data
from logger import get_logger

logger = get_logger(__name__)

MODEL_PATH         = "models/lstm_wqi_model.keras"
SCALER_PATH        = "models/scaler.pkl"
TARGET_SCALER_PATH = "models/target_scaler.pkl"
DATA_PATH          = "data/water_quality.csv"
SEQ_LENGTH         = 24
DELAY_SECS         = 0.5
N_STEPS            = 50
WQI_THRESHOLD      = 70.0
FEATURE_COLS = ["pH","Turbidity","Dissolved_Oxygen","Temperature","Conductivity","BOD","COD"]
TARGET_COL   = "WQI"


def _status_label(wqi):
    return "SAFE  " if wqi >= WQI_THRESHOLD else "UNSAFE"


def _color_bar(wqi, width=30):
    filled = int((wqi / 100) * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {wqi:6.2f}"


def run_simulation():
    for path in [MODEL_PATH, SCALER_PATH, TARGET_SCALER_PATH]:
        if not os.path.exists(path):
            logger.error("Asset not found: '%s'. Run train.py first.", path)
            sys.exit(1)
    if not os.path.exists(DATA_PATH):
        logger.error("Dataset not found: '%s'. Please provide a real CSV.", DATA_PATH)
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    df = load_data(DATA_PATH)
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()
    for col in FEATURE_COLS:
        if df[col].isna().any():
            df[col].fillna(df[col].mean(), inplace=True)

    # Compute WQI if not present
    if TARGET_COL not in df.columns:
        from wqi import calculate_scientific_wqi
        df[TARGET_COL] = calculate_scientific_wqi(df)

    features = df[FEATURE_COLS].values
    targets = df[TARGET_COL].values
    scaled_features = scaler.transform(features)

    train_size = int(len(scaled_features) * 0.8)
    start_idx = train_size

    print("\n" + "=" * 70)
    print("  Real-Time Water Quality Prediction Simulation")
    print(f"  {'Step':<6} {'Pred WQI':>8}  {'Actual':>7}  {'Status':<10}  Bar")
    print("=" * 70)

    errors = []
    for step in range(N_STEPS):
        window_start = start_idx + step
        window_end = window_start + SEQ_LENGTH
        if window_end >= len(scaled_features):
            break

        window = scaled_features[window_start:window_end]
        lstm_input = np.expand_dims(window, axis=0)
        raw_pred = model.predict(lstm_input, verbose=0)
        pred_wqi = float(np.clip(target_scaler.inverse_transform(raw_pred)[0][0], 0, 100))
        actual_wqi = float(targets[window_end])
        error = abs(pred_wqi - actual_wqi)
        errors.append(error)

        status = _status_label(pred_wqi)
        bar = _color_bar(pred_wqi)
        print(f"  {step+1:<6} {pred_wqi:>8.2f}  {actual_wqi:>7.2f}  {status:<10}  {bar}")
        time.sleep(DELAY_SECS)

    mean_err = np.mean(errors) if errors else float("nan")
    print("\n" + "=" * 70)
    print(f"  Simulation complete. Steps: {len(errors)} | Mean MAE: {mean_err:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_simulation()
