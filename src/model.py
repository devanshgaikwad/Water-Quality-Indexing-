"""
model.py
--------
LSTM model architecture for Water Quality Index (WQI) prediction.

Why LSTM over static models (Random Forest, Linear Regression)?
    - Water quality is *temporal*: readings at T influence quality at T+1.
    - LSTMs learn gated memory cells that capture trends, seasonal cycles,
      and gradual degradation patterns across the 24-hour sliding window.
    - Static models treat each reading independently — they cannot detect
      "turbidity has been rising for 6 hours" because they have no memory.
    - LSTM's forget gate filters transient sensor noise (bubble spikes,
      calibration jitter) that would fool instantaneous models.

Architecture:
    - Stacked LSTM (64 → 32 units)
    - L2 regularisation on all trainable layers
    - Dropout(0.2) between LSTM layers
    - Dense head for regression (output = single WQI value, linear)
    - Huber loss (robust to outliers in water data)

Usage:
    from model import build_lstm_model
    model = build_lstm_model(input_shape=(24, 7))
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


def build_lstm_model(
    input_shape: tuple,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dropout_rate: float = 0.2,
    l2_alpha: float = 0.0001,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Builds and compiles a stacked LSTM model for WQI regression.

    Args:
        input_shape:   (seq_length, num_features) -- e.g. (24, 7).
        lstm_units_1:  Number of units in the first LSTM layer.
        lstm_units_2:  Number of units in the second LSTM layer.
        dropout_rate:  Dropout probability after each LSTM layer.
        l2_alpha:      L2 regularisation strength.
        learning_rate: Learning rate for the Adam optimiser.

    Returns:
        Compiled Keras Sequential model.
    """
    reg = tf.keras.regularizers.l2(l2_alpha)

    model = Sequential(
        [
            Input(shape=input_shape),

            # -- First LSTM layer --
            LSTM(lstm_units_1, return_sequences=True, kernel_regularizer=reg),
            Dropout(dropout_rate),

            # -- Second LSTM layer --
            LSTM(lstm_units_2, return_sequences=False, kernel_regularizer=reg),

            # -- Fully-connected hidden layer --
            Dense(16, activation="relu", kernel_regularizer=reg),

            # -- Output (linear for regression) --
            Dense(1, activation="linear"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
        metrics=["mae"],
    )

    return model


if __name__ == "__main__":
    # Quick build test
    m = build_lstm_model(input_shape=(24, 7))
    m.summary()
