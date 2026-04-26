"""
train.py — Training script for the Water Quality LSTM model.

Pipeline:
    1. Load REAL dataset (no synthetic generation)
    2. Preprocess: clean -> compute WQI -> scale -> T+1 sequences -> chronological split
    3. Run baseline models (Linear Regression + Random Forest) for comparison
    4. Build stacked LSTM model
    5. Train with EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    6. shuffle=False (strict time-series integrity)
    7. Evaluate on HELD-OUT test set
    8. Walk-forward validation (temporal cross-validation)
    9. Permutation feature importance
    10. Manual extreme-value sensitivity test
    11. Concept drift note for interview
"""
import os, sys, numpy as np, tensorflow as tf, joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_data, preprocess_data
from model import build_lstm_model
from evaluate import full_evaluation, walk_forward_validation
from logger import get_logger

logger = get_logger(__name__)

DATA_PATH          = "data/water_quality.csv"
MODEL_PATH         = "models/lstm_wqi_model.keras"
SCALER_PATH        = "models/scaler.pkl"
TARGET_SCALER_PATH = "models/target_scaler.pkl"
OUTPUT_DIR         = "models"

FEATURE_COLS = ["pH","Turbidity","Dissolved_Oxygen","Temperature","Conductivity","BOD","COD"]
TARGET_COL   = "WQI"
SEQ_LENGTH   = 24
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
EPOCHS       = 100
BATCH_SIZE   = 32
PATIENCE     = 15


def run_baselines(X_train, X_test, y_train_orig, y_test_orig):
    """Run Linear Regression + Random Forest baselines for LSTM comparison."""
    # Use last timestep features (most recent reading) for static models
    X_tr_flat = X_train[:, -1, :]
    X_te_flat = X_test[:, -1, :]

    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_tr_flat, y_train_orig)
    lr_pred = lr.predict(X_te_flat)
    lr_rmse = float(np.sqrt(mean_squared_error(y_test_orig, lr_pred)))
    lr_r2 = float(r2_score(y_test_orig, lr_pred))
    results["LinearRegression"] = {"RMSE": lr_rmse, "R2": lr_r2}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr_flat, y_train_orig)
    rf_pred = rf.predict(X_te_flat)
    rf_rmse = float(np.sqrt(mean_squared_error(y_test_orig, rf_pred)))
    rf_r2 = float(r2_score(y_test_orig, rf_pred))
    results["RandomForest"] = {"RMSE": rf_rmse, "R2": rf_r2}

    return results


def validate_model_sensitivity(model, feature_scaler, target_scaler, seq_length=24):
    """Tests model with extreme inputs to verify different inputs -> different outputs."""
    from wqi import calculate_current_wqi_from_values

    test_cases = [
        {"pH":7.0,"Turbidity":0.0,"Dissolved_Oxygen":14.6,"Temperature":25.0,
         "Conductivity":0.0,"BOD":0.0,"COD":0.0,"label":"Pristine water"},
        {"pH":7.5,"Turbidity":3.0,"Dissolved_Oxygen":9.0,"Temperature":20.0,
         "Conductivity":300.0,"BOD":5.0,"COD":10.0,"label":"Optimal water"},
        {"pH":5.0,"Turbidity":25.0,"Dissolved_Oxygen":4.0,"Temperature":30.0,
         "Conductivity":600.0,"BOD":20.0,"COD":40.0,"label":"Moderate pollution"},
        {"pH":3.0,"Turbidity":50.0,"Dissolved_Oxygen":2.0,"Temperature":35.0,
         "Conductivity":800.0,"BOD":40.0,"COD":80.0,"label":"Very poor water"},
        {"pH":14.0,"Turbidity":90.0,"Dissolved_Oxygen":0.5,"Temperature":40.0,
         "Conductivity":950.0,"BOD":80.0,"COD":150.0,"label":"Extreme pollution"},
    ]

    print("\n" + "=" * 80)
    print("  SENSITIVITY TEST: Verifying LSTM produces different outputs")
    print("=" * 80)

    predictions = []
    for tc in test_cases:
        raw = np.array([[tc["pH"],tc["Turbidity"],tc["Dissolved_Oxygen"],
                         tc["Temperature"],tc["Conductivity"],tc["BOD"],tc["COD"]]],
                       dtype=np.float32)
        scaled = feature_scaler.transform(raw)
        window = np.tile(scaled, (seq_length, 1))
        lstm_in = np.expand_dims(window, axis=0)
        raw_pred = model.predict(lstm_in, verbose=0)
        predicted_wqi = float(np.clip(target_scaler.inverse_transform(raw_pred)[0][0], 0, 100))

        scientific_wqi = calculate_current_wqi_from_values(
            tc["pH"],tc["Turbidity"],tc["Dissolved_Oxygen"],
            tc["Temperature"],tc["Conductivity"],tc["BOD"],tc["COD"]
        )

        predictions.append(predicted_wqi)
        status = "Safe" if predicted_wqi >= 70 else "Unsafe"
        print(f"  [{tc['label']}]")
        print(f"    Scientific WQI: {scientific_wqi:.2f} | LSTM Predicted: {predicted_wqi:.2f} ({status})")
        print(f"    Delta: {predicted_wqi - scientific_wqi:+.2f}")
        print()

    unique_preds = len(set(round(p, 1) for p in predictions))
    if unique_preds >= 3:
        print(f"  [PASS] {unique_preds} distinct predictions for {len(test_cases)} inputs.")
    else:
        print(f"  [WARN] Only {unique_preds} distinct predictions.")
    print("=" * 80)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load & preprocess
    logger.info("Loading and preprocessing REAL dataset ...")
    df = load_data(DATA_PATH)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler = preprocess_data(
        df, feature_cols=FEATURE_COLS, target_col=TARGET_COL, seq_length=SEQ_LENGTH,
        train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
        scaler_path=SCALER_PATH, target_scaler_path=TARGET_SCALER_PATH,
    )

    logger.info("X_train: %s | y_train: %s", X_train.shape, y_train.shape)
    logger.info("X_val  : %s | y_val  : %s", X_val.shape, y_val.shape)
    logger.info("X_test : %s | y_test : %s", X_test.shape, y_test.shape)

    # Original-scale targets for baselines
    y_train_orig = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # 2. Baseline models
    logger.info("Running baseline models (LR + RF) for comparison ...")
    baselines = run_baselines(X_train, X_test, y_train_orig, y_test_orig)

    print("\n" + "=" * 60)
    print("  BASELINE MODEL COMPARISON")
    print("=" * 60)
    for name, m in baselines.items():
        print(f"  {name:20s} | RMSE: {m['RMSE']:.4f} | R2: {m['R2']:.4f}")
    print("=" * 60)

    # 3. Build LSTM model
    logger.info("Building LSTM model ...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    model.summary(print_fn=logger.info)

    # 4. Train with shuffle=False
    logger.info("Starting training (epochs=%d, batch=%d, patience=%d, shuffle=False) ...",
                EPOCHS, BATCH_SIZE, PATIENCE)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks,
        verbose=1, shuffle=False,
    )

    logger.info("Training complete. Best model saved -> '%s'.", MODEL_PATH)

    # 5. Evaluate on test set
    logger.info("Running evaluation on HELD-OUT test set ...")
    y_pred_scaled = model.predict(X_test)
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled).flatten()

    y_train_pred_scaled = model.predict(X_train)
    y_train_pred_orig = target_scaler.inverse_transform(y_train_pred_scaled).flatten()

    all_metrics = full_evaluation(
        y_true=y_test_orig, y_pred=y_pred_original, history=history,
        output_dir=OUTPUT_DIR, model=model, X_val=X_val, y_val=y_val,
        target_scaler=target_scaler,
    )

    # 6. Train vs Test comparison
    train_rmse = float(np.sqrt(mean_squared_error(y_train_orig, y_train_pred_orig)))
    train_r2 = float(r2_score(y_train_orig, y_train_pred_orig))

    print("\n" + "=" * 60)
    print("  FULL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<20s} | {'RMSE':>10s} | {'R2':>10s}")
    print(f"  {'-'*20} | {'-'*10} | {'-'*10}")
    for name, m in baselines.items():
        print(f"  {name:<20s} | {m['RMSE']:>10.4f} | {m['R2']:>10.4f}")
    print(f"  {'LSTM (train)':<20s} | {train_rmse:>10.4f} | {train_r2:>10.4f}")
    print(f"  {'LSTM (test)':<20s} | {all_metrics['RMSE']:>10.4f} | {all_metrics['R2']:>10.4f}")
    print("=" * 60)

    # 7. Walk-forward validation
    logger.info("Running walk-forward validation ...")
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    wf_results = walk_forward_validation(X_all, y_all, target_scaler, build_lstm_model, n_splits=3)

    if wf_results:
        print("\n" + "=" * 60)
        print("  WALK-FORWARD VALIDATION (Temporal CV)")
        print("=" * 60)
        for r in wf_results:
            print(f"  Fold {r['fold']}: train={r['train_size']}, test={r['test_size']} "
                  f"| RMSE={r['RMSE']:.4f} | R2={r['R2']:.4f}")
        avg_rmse = np.mean([r["RMSE"] for r in wf_results])
        avg_r2 = np.mean([r["R2"] for r in wf_results])
        print(f"  {'Average':<20s} | {avg_rmse:>10.4f} | {avg_r2:>10.4f}")
        print("=" * 60)

    # 8. Sensitivity validation
    validate_model_sensitivity(model, feature_scaler, target_scaler, SEQ_LENGTH)

    # 9. Concept drift note
    print("\n" + "=" * 60)
    print("  CONCEPT DRIFT AWARENESS")
    print("=" * 60)
    print("  This model was trained on a fixed dataset. In production:")
    print("  - Monitor prediction error over time (drift detection)")
    print("  - If rolling RMSE > 2x training RMSE -> trigger retraining")
    print("  - Consider online learning or periodic batch retraining")
    print("  - Track feature distribution shifts (KS test, PSI)")
    print("=" * 60)

    print(f"\n  Plots saved to '{OUTPUT_DIR}/'")
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
