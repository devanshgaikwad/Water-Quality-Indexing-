"""
dashboard.py — Streamlit Water Quality Monitoring Dashboard
------------------------------------------------------------
Clean, interview-ready dual-output dashboard.

Layout:
    TOP:    Title bar
    LEFT:   Current WQI (big number) + Safe/Unsafe status
    RIGHT:  Predicted WQI (big number) + Trend arrow + Delta
    STRIP:  Confidence | Explanation | Alert
    CHART:  Time vs WQI (current line, predicted line, threshold)
    SIDEBAR: Manual Input (7 sliders) + Simulation + Buffer status
"""
import streamlit as st
import pandas as pd
import numpy as np
import time, os, sys, joblib
import tensorflow as tf
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from wqi import calculate_current_wqi_from_values, VALID_RANGES

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Water Quality Prediction", page_icon="💧", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d0d24 100%);
        font-family: 'Inter', sans-serif;
    }
    .big-title {
        text-align: center; font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #4FC3F7, #00E676);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem; letter-spacing: -0.5px;
    }
    .subtitle {
        text-align: center; color: #78909C; font-size: 0.9rem;
        margin-bottom: 1.5rem; font-weight: 400;
    }
    .wqi-card {
        background: rgba(20, 20, 40, 0.9); backdrop-filter: blur(16px);
        padding: 28px 24px; border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        text-align: center; min-height: 200px;
    }
    .wqi-label { font-size: 0.85rem; color: #90A4AE; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .wqi-value { font-size: 3.5rem; font-weight: 800; margin: 8px 0;
        line-height: 1; }
    .wqi-safe { color: #00E676; }
    .wqi-unsafe { color: #FF5252; }
    .wqi-status { font-size: 1rem; font-weight: 700; margin-top: 4px; }
    .trend-badge { display: inline-block; padding: 6px 16px; border-radius: 20px;
        font-size: 1.1rem; font-weight: 700; margin-top: 8px; }
    .trend-up { background: rgba(0,230,118,0.15); color: #00E676; }
    .trend-down { background: rgba(255,82,82,0.15); color: #FF5252; }
    .trend-stable { background: rgba(255,213,79,0.15); color: #FFD54F; }
    .info-strip {
        background: rgba(20, 20, 40, 0.7); backdrop-filter: blur(10px);
        padding: 16px 24px; border-radius: 12px; margin-top: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        display: flex; gap: 32px; align-items: center; flex-wrap: wrap;
    }
    .info-item { color: #B0BEC5; font-size: 0.9rem; }
    .info-label { color: #607D8B; font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.5px; display: block; margin-bottom: 2px; }
    .alert-banner {
        background: rgba(255, 82, 82, 0.12); border: 1px solid rgba(255,82,82,0.4);
        border-radius: 10px; padding: 14px 20px; margin-top: 16px;
        color: #FF5252; font-weight: 700; font-size: 0.95rem;
    }
    .warmup-banner {
        background: rgba(255, 213, 79, 0.12); border: 1px solid rgba(255,213,79,0.3);
        border-radius: 10px; padding: 14px 20px; margin-top: 16px;
        color: #FFD54F; font-weight: 600; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">💧 Real-Time Water Quality Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dual System: Scientific WQI (Current) + LSTM Forecast (Future T+1) <br> <b>Score Scale: 0-100 (Higher is better. ≥ 70 is Safe, < 70 is Unsafe)</b></div>', unsafe_allow_html=True)

# ── Config ───────────────────────────────────────────────────────────────────
SEQ_LENGTH = 24
WQI_THRESHOLD = 70.0
FEATURE_NAMES = ["pH","Turbidity","Dissolved_Oxygen","Temperature","Conductivity","BOD","COD"]


@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "lstm_wqi_model.keras"))
        scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
        target_scaler = joblib.load(os.path.join(BASE_DIR, "models", "target_scaler.pkl"))
        return model, scaler, target_scaler
    except Exception as e:
        st.error(f"Failed to load assets: {e}")
        return None, None, None

model, scaler, target_scaler = load_assets()


def explain_prediction_mean(model_obj, scaled_input, base_wqi):
    """Mean-replacement explainability — keeps inputs in-distribution."""
    impacts = {}
    for i, name in enumerate(FEATURE_NAMES):
        perturbed = scaled_input.copy()
        perturbed[:, i] = np.mean(scaled_input[:, i])
        pert_pred = model_obj.predict(np.expand_dims(perturbed, 0), verbose=0)
        pert_wqi = float(target_scaler.inverse_transform(pert_pred)[0][0])
        impacts[name] = abs(base_wqi - pert_wqi)
    total = sum(impacts.values()) + 1e-9
    pct = {k: (v / total) * 100 for k, v in impacts.items()}
    sorted_f = sorted(pct.items(), key=lambda x: -x[1])
    
    top1 = sorted_f[0]
    top2 = sorted_f[1]
    
    if base_wqi >= 70:
        return f"Water is expected to be Safe. Score is mainly maintained by {top1[0]} ({top1[1]:.1f}%) and {top2[0]} ({top2[1]:.1f}%)."
    else:
        return f"Water is expected to be Unsafe. Degradation is heavily driven by {top1[0]} ({top1[1]:.1f}%) and {top2[0]} ({top2[1]:.1f}%)."


def predict_lstm(feature_array_24):
    if model is None or scaler is None or target_scaler is None:
        raise Exception("Model not loaded. Run python src/train.py first.")

    # Clip to physical bounds
    for i, name in enumerate(FEATURE_NAMES):
        lo, hi = VALID_RANGES[name]
        feature_array_24[:, i] = np.clip(feature_array_24[:, i], lo, hi)

    scaled = scaler.transform(feature_array_24)
    lstm_in = np.expand_dims(scaled, axis=0)
    raw_pred = model.predict(lstm_in, verbose=0)
    predicted_wqi = float(np.clip(target_scaler.inverse_transform(raw_pred)[0][0], 0, 100))
    explanation = explain_prediction_mean(model, scaled, predicted_wqi)
    return {"predicted_wqi": round(predicted_wqi, 2), "explanation": explanation}


# ── Load Historical Data ─────────────────────────────────────────────────────
@st.cache_data
def load_historical_data():
    data_path = os.path.join(BASE_DIR, "data", "water_quality.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df.tail(200).reset_index(drop=True)
    return None

df_hist = load_historical_data()
if df_hist is None:
    st.warning("Historical data not found. Run `python src/train.py` first.")
    st.stop()

# ── Session State ─────────────────────────────────────────────────────────────
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "buffer" not in st.session_state:
    st.session_state.buffer = df_hist.iloc[0:SEQ_LENGTH][FEATURE_NAMES].values.tolist()
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "prev_predicted_wqi" not in st.session_state:
    st.session_state.prev_predicted_wqi = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Controls")
simulate = st.sidebar.button("▶️  Simulate Next Hour")
auto_simulate = st.sidebar.checkbox("⏩  Auto Simulate (2s)", value=False)

st.sidebar.markdown("---")
if model is not None:
    st.sidebar.success("🟢 LSTM Model Loaded")
else:
    st.sidebar.error("🔴 Model Offline")

buf_len = len(st.session_state.buffer)
st.sidebar.metric("Buffer Status", f"{buf_len}/{SEQ_LENGTH}")
if buf_len < SEQ_LENGTH:
    st.sidebar.warning(f"Warming up... {SEQ_LENGTH - buf_len} more readings needed")
else:
    st.sidebar.info("Buffer full — LSTM predictions active")

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Manual Input")
st.sidebar.caption("Append a new reading to the sliding window buffer:")
manual_ph = st.sidebar.slider("pH", 0.0, 14.0, 7.5, 0.1)
manual_turb = st.sidebar.slider("Turbidity (NTU)", 0.0, 100.0, 3.0, 0.5)
manual_do = st.sidebar.slider("Dissolved O₂ (mg/L)", 0.0, 20.0, 8.0, 0.1)
manual_temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 20.0, 0.5)
manual_cond = st.sidebar.slider("Conductivity (µS/cm)", 0.0, 1000.0, 300.0, 5.0)
manual_bod = st.sidebar.slider("BOD (mg/L)", 0.0, 100.0, 5.0, 0.5)
manual_cod = st.sidebar.slider("COD (mg/L)", 0.0, 200.0, 10.0, 1.0)

if st.sidebar.button("🧪 Predict (Append to Buffer)"):
    new_reading = [manual_ph, manual_turb, manual_do, manual_temp, manual_cond, manual_bod, manual_cod]
    st.session_state.buffer.append(new_reading)
    if len(st.session_state.buffer) > SEQ_LENGTH:
        st.session_state.buffer = st.session_state.buffer[-SEQ_LENGTH:]

# ── Main Layout ──────────────────────────────────────────────────────────────
buffer_array = np.array(st.session_state.buffer, dtype=np.float32)
latest_reading = buffer_array[-1]

# Compute current WQI (always available)
current_wqi = calculate_current_wqi_from_values(
    float(latest_reading[0]), float(latest_reading[1]), float(latest_reading[2]),
    float(latest_reading[3]), float(latest_reading[4]),
    float(latest_reading[5]), float(latest_reading[6])
)
current_status = "Safe" if current_wqi >= WQI_THRESHOLD else "Unsafe"
current_color = "wqi-safe" if current_wqi >= WQI_THRESHOLD else "wqi-unsafe"

# Check if buffer is full for LSTM prediction
is_warming_up = len(buffer_array) < SEQ_LENGTH

if not is_warming_up:
    try:
        lstm_input = buffer_array[-SEQ_LENGTH:]
        result = predict_lstm(lstm_input)
        predicted_wqi = result["predicted_wqi"]
        explanation = result["explanation"]

        delta = predicted_wqi - current_wqi
        if delta > 2:
            trend_text, trend_arrow, trend_class = "Improving", "↑", "trend-up"
        elif delta < -2:
            trend_text, trend_arrow, trend_class = "Degrading", "↓", "trend-down"
        else:
            trend_text, trend_arrow, trend_class = "Stable", "→", "trend-stable"

        pred_status = "Safe" if predicted_wqi >= WQI_THRESHOLD else "Unsafe"
        pred_color = "wqi-safe" if predicted_wqi >= WQI_THRESHOLD else "wqi-unsafe"

        # Confidence
        dist = abs(predicted_wqi - WQI_THRESHOLD)
        recent_preds = [p["Predicted_WQI"] for p in st.session_state.predictions[-5:]] if st.session_state.predictions else [predicted_wqi]
        variance = float(np.std(recent_preds)) if len(recent_preds) > 1 else 0.0
        if dist > 20 and variance < 5:
            confidence = "High"
        elif dist > 10:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Anomaly
        anomaly_msg = None
        if predicted_wqi < WQI_THRESHOLD:
            anomaly_msg = f"⚠️ Pollution Risk: Predicted WQI ({predicted_wqi:.1f}) below safety threshold ({WQI_THRESHOLD})"
        if st.session_state.prev_predicted_wqi is not None:
            drop_pct = ((st.session_state.prev_predicted_wqi - predicted_wqi) / (st.session_state.prev_predicted_wqi + 1e-9)) * 100
            if drop_pct > 15:
                anomaly_msg = f"🚨 RAPID DROP: WQI dropped {drop_pct:.1f}% from previous reading!"

        st.session_state.prev_predicted_wqi = predicted_wqi

        # Record
        ts_val = time.strftime("%H:%M:%S")
        st.session_state.predictions.append({
            "time": ts_val, "Current_WQI": current_wqi,
            "Predicted_WQI": predicted_wqi, "Delta": delta,
        })
        if len(st.session_state.predictions) > 100:
            st.session_state.predictions = st.session_state.predictions[-100:]

        # ── DISPLAY: Two cards side by side ──────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f'''
            <div class="wqi-card">
                <div class="wqi-label">Current WQI (Scientific)</div>
                <div class="wqi-value {current_color}">{current_wqi:.1f}</div>
                <div class="wqi-status {current_color}">● {current_status}</div>
                <div style="color:#607D8B;font-size:0.75rem;margin-top:12px;">Based on BIS IS:10500 formula</div>
            </div>
            ''', unsafe_allow_html=True)

        with col_right:
            st.markdown(f'''
            <div class="wqi-card">
                <div class="wqi-label">Predicted WQI (LSTM T+1)</div>
                <div class="wqi-value {pred_color}">{predicted_wqi:.1f}</div>
                <div class="wqi-status {pred_color}">● {pred_status}</div>
                <div class="{trend_class} trend-badge">{trend_arrow} {delta:+.1f} {trend_text}</div>
            </div>
            ''', unsafe_allow_html=True)

        # ── Info Strip ───────────────────────────────────────────────────
        st.markdown(f'''
        <div class="info-strip">
            <div class="info-item"><span class="info-label">Confidence</span>{confidence}</div>
            <div class="info-item"><span class="info-label">Explanation</span>{explanation}</div>
            <div class="info-item"><span class="info-label">Buffer</span>{len(st.session_state.buffer)}/{SEQ_LENGTH}</div>
        </div>
        ''', unsafe_allow_html=True)

        # ── Alert Banner (only when triggered) ───────────────────────────
        if anomaly_msg:
            st.markdown(f'<div class="alert-banner">{anomaly_msg}</div>', unsafe_allow_html=True)

        # ── Chart ────────────────────────────────────────────────────────
        if len(st.session_state.predictions) > 1:
            pred_df = pd.DataFrame(st.session_state.predictions)
            fig, ax = plt.subplots(figsize=(12, 4))

            ax.plot(pred_df["Current_WQI"], linewidth=2, color="#4FC3F7",
                    label="Scientific WQI (Current)", marker="", linestyle="-")
            ax.plot(pred_df["Predicted_WQI"], linewidth=2.5, color="#00E676",
                    label="LSTM Predicted (T+1)", marker="o", markersize=3)
            ax.axhline(y=WQI_THRESHOLD, color="#FF5252", linestyle="--",
                       linewidth=1.5, label=f"Safety Threshold ({WQI_THRESHOLD})")

            ax.fill_between(range(len(pred_df)), pred_df["Predicted_WQI"], WQI_THRESHOLD,
                            where=(pred_df["Predicted_WQI"] >= WQI_THRESHOLD),
                            interpolate=True, color="#00E676", alpha=0.08)
            ax.fill_between(range(len(pred_df)), pred_df["Predicted_WQI"], WQI_THRESHOLD,
                            where=(pred_df["Predicted_WQI"] < WQI_THRESHOLD),
                            interpolate=True, color="#FF5252", alpha=0.08)

            ax.set_ylim(0, 105)
            ax.set_ylabel("WQI Score", color="white", fontsize=10)
            ax.set_xlabel("Time", color="white", fontsize=10)
            ax.set_title("WQI Timeline: Current vs Predicted", color="white", fontweight="bold", fontsize=12)
            ax.tick_params(colors="white")
            ax.grid(color="white", alpha=0.08)
            fig.patch.set_facecolor("#0a0a1a")
            ax.set_facecolor("#0a0a1a")
            for spine in ax.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.1))
            ax.legend(loc="lower right", fontsize=8, facecolor="#1a1a3e", edgecolor="none",
                      labelcolor="white")
            st.pyplot(fig)
            plt.close(fig)

    except Exception as e:
        st.error(f"Prediction error: {e}")

else:
    # Warming up state
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f'''
        <div class="wqi-card">
            <div class="wqi-label">Current WQI (Scientific)</div>
            <div class="wqi-value {current_color}">{current_wqi:.1f}</div>
            <div class="wqi-status {current_color}">● {current_status}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col_right:
        st.markdown(f'''
        <div class="wqi-card">
            <div class="wqi-label">Predicted WQI (LSTM T+1)</div>
            <div class="wqi-value" style="color:#78909C;">—</div>
            <div style="color:#FFD54F;font-size:0.9rem;margin-top:8px;">Warming up...</div>
        </div>
        ''', unsafe_allow_html=True)

    progress = len(buffer_array) / SEQ_LENGTH
    st.markdown(f'''
    <div class="warmup-banner">
        🔄 Warming up: {len(buffer_array)}/{SEQ_LENGTH} readings collected.
        LSTM predictions will begin after {SEQ_LENGTH - len(buffer_array)} more readings.
    </div>
    ''', unsafe_allow_html=True)
    st.progress(progress)

# ── Educational Section ──────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📖 How this system works (Interview & Study Guide)"):
    st.markdown("""
    ### 🧠 The Dual Architecture
    This system runs two parallel calculations to provide a complete picture of water quality:
    
    1. **Current WQI (Scientific Baseline):** 
       - Calculates the Water Quality Index *right now* using the standard **BIS IS:10500 weighted arithmetic formula**. 
       - It compares current sensor readings (pH, Turbidity, DO, etc.) against established ideal values and safety limits.
       
    2. **Predicted WQI (LSTM Forecast):**
       - Predicts the Water Quality Index **one hour into the future (T+1)**.
       - Powered by a **Stacked LSTM (Long Short-Term Memory) Neural Network**.
       - **Why LSTM?** Static models (like Random Forest) only look at the current instantaneous snapshot. Water quality degrades sequentially. An LSTM Remembers the past 24 hours of data. It can learn that *"turbidity has been steadily rising for 6 hours"*, allowing it to predict a pollution event *before* it actually happens.
    
    ### ⚙️ Under the Hood Features
    - **Mean-Replacement Explainability:** To explain *why* the AI made its prediction, we test how much each sensor influenced the outcome. Instead of zeroing out sensors (which creates impossible, "out-of-distribution" data), the system replaces a reading with its 24-hour average to scientifically isolate its impact.
    - **Data Quality Defense Layer:** Real-world IoT sensors glitch. This API actively intercepts and clips impossible out-of-bounds readings (like a pH of 15) to its physical limits to protect the AI model from failing.
    - **Walk-Forward Validation:** The model wasn't just tested on a random split of data. It was evaluated using temporal cross-validation—training on the past and testing on the future—proving it can handle the realistic passage of time.
    """)

# ── Simulation ───────────────────────────────────────────────────────────────
if simulate or auto_simulate:
    st.session_state.current_idx += 1
    if st.session_state.current_idx >= len(df_hist):
        st.session_state.current_idx = 0
        st.session_state.predictions = []
    new_row = df_hist.iloc[st.session_state.current_idx][FEATURE_NAMES].values.tolist()
    st.session_state.buffer.append(new_row)
    if len(st.session_state.buffer) > SEQ_LENGTH:
        st.session_state.buffer = st.session_state.buffer[-SEQ_LENGTH:]
    if auto_simulate:
        time.sleep(2)
        st.rerun()
