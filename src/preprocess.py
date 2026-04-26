"""
preprocess.py
-------------
Data preprocessing pipeline for the Water Quality LSTM project.

Responsibilities:
    1. Load real CSV data (NO synthetic generation)
    2. Handle missing values (forward-fill -> backward-fill -> column-mean)
    3. Compute scientific WQI target using BIS formula
    4. Normalise features AND target using separate MinMaxScalers
    5. Generate sliding-window sequences for LSTM input (T+1 prediction)
    6. Strict chronological 70/15/15 train/val/test split (NO shuffling)
    7. Save/load the fitted scalers for inference
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

from logger import get_logger

logger = get_logger(__name__)

FEATURE_COLS = [
    "pH", "Turbidity", "Dissolved_Oxygen", "Temperature",
    "Conductivity", "BOD", "COD",
]


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------

def load_data(filepath: str = "data/water_quality.csv") -> pd.DataFrame:
    """Load real water quality CSV. Exits if file not found."""
    if not os.path.exists(filepath):
        logger.error("Dataset not found at '%s'. Please provide a real dataset.", filepath)
        sys.exit(1)

    df = pd.read_csv(filepath)
    logger.info("Loaded dataset: %d rows x %d cols from '%s'", len(df), len(df.columns), filepath)
    return df


# ---------------------------------------------------------------------------
# 2. Handle Missing Values
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Fills missing values using: forward-fill -> backward-fill -> column-mean.
    Also removes duplicate rows.
    """
    missing_before = df[feature_cols].isna().sum().sum()
    if missing_before > 0:
        logger.info("Missing values detected: %d total across %s", missing_before, feature_cols)

    df[feature_cols] = df[feature_cols].ffill().bfill()

    for col in feature_cols:
        if df[col].isna().any():
            col_mean = df[col].mean()
            df[col].fillna(col_mean, inplace=True)
            logger.warning("Column '%s' had persistent NaN -- filled with mean (%.4f).", col, col_mean)

    dup_count = df.duplicated(subset=feature_cols).sum()
    if dup_count > 0:
        logger.info("Removing %d duplicate rows.", dup_count)
        df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)

    missing_after = df[feature_cols].isna().sum().sum()
    logger.info("Missing values after imputation: %d", missing_after)
    return df


# ---------------------------------------------------------------------------
# 3. Create Sliding-Window Sequences (T+1 prediction)
# ---------------------------------------------------------------------------

def create_sequences(features: np.ndarray, targets: np.ndarray, seq_length: int = 24) -> tuple:
    """
    Creates sequences where X = [t, t+1, ..., t+seq_length-1] and y = target at t+seq_length.
    This is TRUE T+1 future prediction.
    """
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(targets[i + seq_length])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    logger.info("Created %d sequences of length %d (T+1 prediction)", len(X), seq_length)
    return X, y


# ---------------------------------------------------------------------------
# 4. Full Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess_data(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = "WQI",
    seq_length: int = 24,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    scaler_path: str = "models/scaler.pkl",
    target_scaler_path: str = "models/target_scaler.pkl",
) -> tuple:
    """
    End-to-end preprocessing: clean -> compute WQI -> scale -> sequence -> split.
    Uses STRICT chronological split (NO shuffling) to prevent data leakage.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Step 1: Handle missing values and duplicates
    df = handle_missing_values(df, feature_cols)

    # Step 2: Compute Scientific WQI as the ground-truth target
    from wqi import calculate_scientific_wqi
    df[target_col] = calculate_scientific_wqi(df)
    logger.info("Computed scientific WQI using BIS standards.")
    logger.info("Target '%s' stats: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
                target_col, df[target_col].mean(), df[target_col].std(),
                df[target_col].min(), df[target_col].max())

    # Step 3: Extract numpy arrays
    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)

    # Step 4: Normalise features to [0, 1]
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(features)

    # Step 5: Normalise target to [0, 1] using SEPARATE scaler
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_targets = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()

    # Save scalers
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(feature_scaler, scaler_path)
    joblib.dump(target_scaler, target_scaler_path)
    logger.info("Feature scaler saved to '%s'", scaler_path)
    logger.info("Target scaler saved to '%s'", target_scaler_path)

    # Step 6: Create T+1 sequences
    X, y = create_sequences(scaled_features, scaled_targets, seq_length)

    # Step 7: Strict chronological split (NO shuffling)
    n_total = len(X)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    # Step 8: Small Gaussian noise augmentation for training robustness
    noise = np.random.normal(0, 0.01, X_train.shape).astype(np.float32)
    X_train = X_train + noise

    logger.info(
        "Data split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%)",
        len(X_train), train_ratio * 100,
        len(X_val), val_ratio * 100,
        len(X_test), (1 - train_ratio - val_ratio) * 100,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler
