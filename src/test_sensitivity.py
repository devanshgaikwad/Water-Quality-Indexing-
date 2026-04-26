import os, numpy as np, pandas as pd, joblib
from tensorflow.keras.models import load_model

SEQ_LENGTH = 24
FEATURE_COLS = ["pH","Turbidity","Dissolved_Oxygen","Temperature","Conductivity","BOD","COD"]


def run_sensitivity_tests():
    print("Loading models and scalers...")
    model = load_model("models/lstm_wqi_model.keras")
    scaler = joblib.load("models/scaler.pkl")
    target_scaler = joblib.load("models/target_scaler.pkl")

    df = pd.read_csv("data/water_quality.csv")
    base_history = df.iloc[-SEQ_LENGTH+1:][FEATURE_COLS].values

    print("\n--- Running 5 Sensitivity Tests ---")

    test_cases = [
        {"name": "1. Pristine Water", "input": [7.5, 0.5, 9.5, 20.0, 200.0, 1.0, 2.0]},
        {"name": "2. High Turbidity", "input": [7.5, 80.0, 9.5, 20.0, 200.0, 1.0, 2.0]},
        {"name": "3. Acidic Water",   "input": [3.0, 0.5, 9.5, 20.0, 200.0, 1.0, 2.0]},
        {"name": "4. Low Oxygen",     "input": [7.5, 0.5, 2.0, 20.0, 200.0, 1.0, 2.0]},
        {"name": "5. Toxic Sludge",   "input": [1.0, 95.0, 0.5, 45.0, 950.0, 80.0, 150.0]},
    ]

    for test in test_cases:
        manual_input = np.array(test["input"])
        seq = np.vstack([base_history, manual_input])
        scaled_seq = scaler.transform(seq)
        lstm_in = np.expand_dims(scaled_seq, axis=0)
        raw_pred = model.predict(lstm_in, verbose=0)
        wqi = float(np.clip(target_scaler.inverse_transform(raw_pred)[0][0], 0, 100))

        print(f"\n{test['name']}")
        print(f"  Raw Input (Last hr) : {manual_input}")
        print(f"  Scaled Input (Last) : {scaled_seq[-1].round(3)}")
        print(f"  Predicted WQI       : {wqi:.2f}")


if __name__ == "__main__":
    run_sensitivity_tests()
