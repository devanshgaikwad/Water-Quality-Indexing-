"""
evaluate.py — Evaluation module for the Water Quality LSTM model.
Provides: regression metrics, classification metrics, walk-forward validation,
permutation feature importance, and all evaluation plots.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from logger import get_logger

logger = get_logger(__name__)
WQI_THRESHOLD = 70.0
FEATURE_NAMES = ["pH","Turbidity","Dissolved_Oxygen","Temperature","Conductivity","BOD","COD"]


def compute_regression_metrics(y_true, y_pred):
    y_pred_flat = y_pred.flatten()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_flat)))
    mae = float(mean_absolute_error(y_true, y_pred_flat))
    r2 = float(r2_score(y_true, y_pred_flat))
    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
    logger.info("Regression: RMSE=%.4f MAE=%.4f R2=%.4f", rmse, mae, r2)
    return metrics


def compute_classification_metrics(y_true, y_pred):
    y_pred_flat = y_pred.flatten()
    y_true_labels = (y_true >= WQI_THRESHOLD).astype(int)
    y_pred_labels = (y_pred_flat >= WQI_THRESHOLD).astype(int)
    metrics = {
        "Accuracy": float(accuracy_score(y_true_labels, y_pred_labels)),
        "Precision": float(precision_score(y_true_labels, y_pred_labels, zero_division=0)),
        "Recall": float(recall_score(y_true_labels, y_pred_labels, zero_division=0)),
        "F1": float(f1_score(y_true_labels, y_pred_labels, zero_division=0)),
    }
    logger.info("Classification (threshold=%.1f): Acc=%.4f P=%.4f R=%.4f F1=%.4f",
                WQI_THRESHOLD, metrics["Accuracy"], metrics["Precision"],
                metrics["Recall"], metrics["F1"])
    return metrics, y_true_labels, y_pred_labels


def plot_training_history(history, save_path="models/training_history.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Training History", fontsize=14, fontweight="bold")
    axes[0].plot(history.history["loss"], label="Train Loss", color="#2196F3")
    axes[0].plot(history.history["val_loss"], label="Val Loss", color="#FF5722", linestyle="--")
    axes[0].set_title("Huber Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    if "mae" in history.history:
        axes[1].plot(history.history["mae"], label="Train MAE", color="#4CAF50")
        axes[1].plot(history.history["val_mae"], label="Val MAE", color="#FF9800", linestyle="--")
        axes[1].set_title("MAE"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Training history plot saved -> %s", save_path)


def plot_predictions(y_true, y_pred, n_points=300, save_path="models/predictions_vs_actual.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_pred_flat = y_pred.flatten()
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("Model Evaluation: Predicted vs Actual WQI", fontsize=14, fontweight="bold")
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(y_true[:n_points], label="Actual WQI", color="#1565C0", linewidth=1.2)
    ax1.plot(y_pred_flat[:n_points], label="Predicted WQI", color="#E53935", linewidth=1.2, alpha=0.8)
    ax1.axhline(y=WQI_THRESHOLD, color="green", linestyle="--", linewidth=1, label=f"Threshold ({WQI_THRESHOLD})")
    ax1.set_title(f"First {n_points} Test Samples"); ax1.set_xlabel("Time Step"); ax1.set_ylabel("WQI")
    ax1.set_ylim(0, 110); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(y_true, y_pred_flat, alpha=0.3, s=10, color="#7B1FA2")
    mn, mx = min(y_true.min(), y_pred_flat.min()), max(y_true.max(), y_pred_flat.max())
    ax2.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect Fit")
    ax2.set_title("Actual vs Predicted Scatter"); ax2.set_xlabel("Actual WQI"); ax2.set_ylabel("Predicted WQI")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax3 = fig.add_subplot(gs[1, 1])
    residuals = y_true - y_pred_flat
    ax3.hist(residuals, bins=50, color="#00897B", edgecolor="white", alpha=0.85)
    ax3.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax3.set_title("Residuals Distribution"); ax3.set_xlabel("Residual"); ax3.set_ylabel("Count"); ax3.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Predictions plot saved -> %s", save_path)


def plot_confusion_matrix(y_true_labels, y_pred_labels, save_path="models/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Unsafe", "Safe"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Safe / Unsafe)")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Confusion matrix saved -> %s", save_path)


def permutation_importance_lstm(model, X_val, y_val, target_scaler, n_repeats=5):
    """Global permutation feature importance — shuffles each feature and measures RMSE increase."""
    logger.info("Calculating Permutation Feature Importance (%d repeats)...", n_repeats)
    base_pred = target_scaler.inverse_transform(model.predict(X_val, verbose=0)).flatten()
    y_orig = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    base_rmse = float(np.sqrt(mean_squared_error(y_orig, base_pred)))
    importances = {}
    for idx, name in enumerate(FEATURE_NAMES):
        scores = []
        for _ in range(n_repeats):
            X_s = X_val.copy()
            flat = X_s[:, :, idx].flatten()
            np.random.shuffle(flat)
            X_s[:, :, idx] = flat.reshape(X_s.shape[0], X_s.shape[1])
            s_pred = target_scaler.inverse_transform(model.predict(X_s, verbose=0)).flatten()
            scores.append(float(np.sqrt(mean_squared_error(y_orig, s_pred))))
        importances[name] = max(np.mean(scores) - base_rmse, 0.0)
    logger.info("Feature Importances (RMSE increase):")
    for n in sorted(importances, key=importances.get, reverse=True):
        logger.info("  %s: +%.4f", n, importances[n])
    return importances


def walk_forward_validation(X_all, y_all, target_scaler, build_model_fn, n_splits=3, initial_train_ratio=0.5):
    """
    Walk-forward validation — trains on past, tests on future window, slides forward.
    Simulates real-world temporal deployment. Superior to single chronological split.
    """
    logger.info("Walk-Forward Validation (%d splits)...", n_splits)
    n = len(X_all)
    init_size = int(n * initial_train_ratio)
    fold_size = (n - init_size) // n_splits
    results = []
    for fold in range(n_splits):
        tr_end = init_size + fold * fold_size
        te_end = min(tr_end + fold_size, n)
        if tr_end >= n or te_end <= tr_end:
            break
        X_tr, y_tr = X_all[:tr_end], y_all[:tr_end]
        X_te, y_te = X_all[tr_end:te_end], y_all[tr_end:te_end]
        if len(X_te) == 0:
            break
        m = build_model_fn(input_shape=(X_tr.shape[1], X_tr.shape[2]))
        m.fit(X_tr, y_tr, epochs=30, batch_size=32, verbose=0, shuffle=False)
        pred = target_scaler.inverse_transform(m.predict(X_te, verbose=0)).flatten()
        actual = target_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mae = float(mean_absolute_error(actual, pred))
        r2 = float(r2_score(actual, pred))
        results.append({"fold": fold+1, "RMSE": rmse, "MAE": mae, "R2": r2,
                         "train_size": len(X_tr), "test_size": len(X_te)})
        logger.info("  Fold %d: train=%d test=%d RMSE=%.4f MAE=%.4f R2=%.4f",
                     fold+1, len(X_tr), len(X_te), rmse, mae, r2)
    if results:
        logger.info("  Walk-Forward Avg: RMSE=%.4f R2=%.4f",
                     np.mean([r["RMSE"] for r in results]), np.mean([r["R2"] for r in results]))
    return results


def full_evaluation(y_true, y_pred, history=None, output_dir="models",
                    model=None, X_val=None, y_val=None, target_scaler=None):
    """Runs the complete evaluation pipeline: metrics + plots + feature importance."""
    reg = compute_regression_metrics(y_true, y_pred)
    cls, yt_lbl, yp_lbl = compute_classification_metrics(y_true, y_pred)
    all_metrics = {**reg, **cls}
    if history is not None:
        plot_training_history(history, save_path=os.path.join(output_dir, "training_history.png"))
    plot_predictions(y_true, y_pred, save_path=os.path.join(output_dir, "predictions_vs_actual.png"))
    plot_confusion_matrix(yt_lbl, yp_lbl, save_path=os.path.join(output_dir, "confusion_matrix.png"))
    if model is not None and X_val is not None and y_val is not None and target_scaler is not None:
        permutation_importance_lstm(model, X_val, y_val, target_scaler)
    logger.info("Full evaluation complete. Plots saved -> '%s/'", output_dir)
    return all_metrics
