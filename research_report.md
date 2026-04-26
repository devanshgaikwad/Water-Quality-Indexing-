# Real-Time Water Quality Monitoring System using LSTM
**Architecture:** Dual System — Scientific WQI (BIS) + Stacked LSTM (T+1 Forecast)

---

## 1. Problem Statement

Water quality monitoring is traditionally limited by sparse manual sampling. This system implements a real-time ML pipeline that:
- Computes **Scientific WQI** (current state) using BIS IS:10500 weighted arithmetic formula
- Predicts **Future WQI** (T+1) using a trained LSTM neural network
- Detects emerging pollution trends before they breach safety thresholds

---

## 2. Scientific Baseline: BIS IS:10500

$$Q_i = \frac{|C_i - V_{ideal}|}{S_i - V_{ideal}} \times 100$$

$$WQI = \frac{\sum (Q_i \times W_i)}{\sum W_i}$$

### Standard Parameters:

| Parameter | Ideal Value (Vi) | BIS Limit (Si) |
|---|---|---|
| pH | 7.0 | 8.5 |
| Turbidity | 0.0 | 5.0 |
| Dissolved_Oxygen | 14.6 | 5.0 |
| Temperature | 25.0 | 35.0 |
| Conductivity | 0.0 | 300.0 |
| BOD | 0.0 | 5.0 |
| COD | 0.0 | 10.0 |


---

## 3. Why LSTM Over Static Models?

| Model | Limitation |
|---|---|
| Linear Regression | Treats each reading independently; no temporal memory |
| Random Forest | Cannot model sequential dependencies |
| Simple RNN | Vanishing gradients on long sequences |
| **LSTM** | **Gated memory captures trends over 24-hour windows** |

**LSTM advantages for water quality:**
- **Temporal smoothing:** Filters transient sensor noise (bubble spikes, calibration jitter)
- **Sequential trends:** Detects gradual deterioration before thresholds are breached
- **Non-linear interactions:** Learns complex pH-DO-temperature relationships
- **Memory:** "Turbidity has been rising for 6 hours" — static models cannot detect this

---

## 4. Model Architecture

```
Input: (24, 7) — 24 hours × 7 sensor features
LSTM(64, return_sequences=True) + L2 + Dropout(0.2)
LSTM(32, return_sequences=False) + L2
Dense(16, relu) + L2
Dense(1, linear)

Loss: Huber (robust to outliers)
Optimizer: Adam (lr=0.001, ReduceLROnPlateau)
Training: shuffle=False, chronological split 70/15/15
Target: T+1 prediction (next hour's WQI)
```

---

## 5. Baseline Comparison

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 7.07 | 0.4688 |
| Random Forest | 3.27 | 0.8863 |
| **LSTM** | **4.26** | **0.8075** |

---

## 6. Evaluation Results (Held-Out Test Set)

| Metric | Value |
|---|---|
| **RMSE** | 4.26 |
| **MAE** | 2.98 |
| **R²** | 0.8075 |

---

## 7. Explainability

### Mean-Replacement Feature Attribution
For each prediction, we replace each feature with its temporal mean and measure prediction change. This keeps inputs in-distribution (unlike zero-out methods).

### Permutation Feature Importance (Global)
During evaluation, we shuffle each feature across the validation set and measure RMSE increase. This is model-agnostic and statistically robust.

> **Interview note:** For instance-level explanations, SHAP (SHapley Additive exPlanations) provides theoretically grounded attributions but is computationally expensive for LSTMs.

---

## 8. Concept Drift Awareness

Water quality distributions change over time (seasonal patterns, new pollution sources). This model:
- Was trained on a fixed dataset snapshot
- Should be monitored for prediction error drift
- **Retraining trigger:** If rolling RMSE exceeds 2× training RMSE
- **Detection methods:** KS test, Population Stability Index (PSI)
- **Mitigation:** Periodic batch retraining or online learning

---

## 9. Limitations

1. **Feature set:** 7 physical/chemical parameters. Heavy metals would improve coverage.
2. **Sensor drift:** Assumes calibrated sensors. Bio-fouling introduces systematic bias.
3. **Distribution shift:** Extreme events outside training range may reduce reliability.
4. **Single location:** Multi-location generalization requires transfer learning.

---

## 10. Walk-Forward Validation

We used temporal cross-validation (walk-forward) to verify the model generalizes across different time periods, not just one specific test window. This simulates real-world deployment at different points in time.

---

**Status:** Pipeline validated. Dual system operational. Research presentation ready.
