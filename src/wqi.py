"""
src/wqi.py
----------
Scientific Water Quality Index (WQI) Calculator
Based on BIS IS:10500 / WHO Drinking Water Standards.

Formula:
    Qi = ((Ci - Vi) / (Si - Vi)) * 100
    WQI = sum(Qi * Wi) / sum(Wi)

Where:
    Ci = observed value of parameter i
    Vi = ideal value of parameter i
    Si = standard permissible limit of parameter i
    Wi = unit weight = k / Si
    k  = proportionality constant = 1 / sum(1/Si)

Scale:
    0-25   = Excellent
    26-50  = Good
    51-75  = Poor
    76-100 = Very Poor
    >100   = Unsuitable

NOTE: Higher WQI = worse water quality (standard convention).
      The dashboard inverts this so 100 = Best, 0 = Worst.
"""
import numpy as np
import pandas as pd


# BIS IS:10500 Standard Limits and Ideal Values
STANDARDS = {
    #                  Si (limit)   Vi (ideal)
    'pH':              (8.5,        7.0),
    'Turbidity':       (5.0,        0.0),
    'Dissolved_Oxygen': (5.0,       14.6),   # Lower bound; ideal is saturated
    'Temperature':     (35.0,       25.0),
    'Conductivity':    (300.0,      0.0),
    'BOD':             (5.0,        0.0),    # Biochemical Oxygen Demand
    'COD':             (10.0,       0.0),    # Chemical Oxygen Demand
}

# Physical bounds for each parameter (used for input validation)
VALID_RANGES = {
    'pH':               (0.0,   14.0),
    'Turbidity':        (0.0,  100.0),
    'Dissolved_Oxygen': (0.0,   20.0),
    'Temperature':      (0.0,   50.0),
    'Conductivity':     (0.0, 1000.0),
    'BOD':              (0.0,  100.0),
    'COD':              (0.0,  200.0),
}


def compute_qi(ci: np.ndarray, si: float, vi: float) -> np.ndarray:
    """
    Compute sub-index Qi for a single parameter.
    Uses absolute deviation to handle bidirectional parameters (e.g., pH).
    pH=5.0 is just as bad as pH=9.0 — both deviate from ideal 7.0.
    """
    denominator = abs(si - vi)
    if denominator < 1e-9:
        return np.zeros_like(ci, dtype=float)
    qi = (np.abs(ci - vi) / denominator) * 100.0
    return np.clip(qi, 0.0, 100.0)


def calculate_scientific_wqi(df: pd.DataFrame) -> pd.Series:
    """
    Computes the scientific WQI for each row in the DataFrame.

    Returns:
        pd.Series with WQI values on an INVERTED scale:
            100 = Pristine (all parameters at ideal)
            0   = Extremely polluted
    """
    # Only use parameters that exist in both STANDARDS and df columns
    available = [p for p in STANDARDS if p in df.columns]

    # Unit weights: Wi = k / Si
    k = 1.0 / sum(1.0 / si for p, (si, _) in STANDARDS.items() if p in available)
    weights = {param: k / STANDARDS[param][0] for param in available}
    sum_w = sum(weights.values())

    wqi_raw = np.zeros(len(df), dtype=float)

    for param in available:
        si, vi = STANDARDS[param]
        ci = df[param].values.astype(float)
        qi = compute_qi(ci, si, vi)
        wqi_raw += qi * weights[param]

    scientific_wqi = wqi_raw / sum_w  # 0 = perfect, 100 = terrible

    # Invert so 100 = Best, 0 = Worst (matches dashboard convention)
    inverted = 100.0 - scientific_wqi
    return pd.Series(np.clip(inverted, 0.0, 100.0), index=df.index)


def calculate_current_wqi_from_values(
    ph: float,
    turb: float,
    do: float,
    temp: float,
    cond: float,
    bod: float = 0.0,
    cod: float = 0.0,
) -> float:
    """
    Convenience function: compute WQI for a single set of sensor readings.
    Returns inverted scale (100 = Best).
    """
    row = pd.DataFrame([{
        'pH': ph,
        'Turbidity': turb,
        'Dissolved_Oxygen': do,
        'Temperature': temp,
        'Conductivity': cond,
        'BOD': bod,
        'COD': cod,
    }])
    return float(calculate_scientific_wqi(row).iloc[0])
