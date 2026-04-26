"""
STEP-BY-STEP DIAGNOSTIC SCRIPT
    Step 1: Verify data quality
    Step 2: Check baselines (Linear Regression, Random Forest)
    Step 3: Target alignment check
    Step 4: Feature analysis + recommendations
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocess import load_data, preprocess_data
import joblib

features = ["pH","Turbidity","Dissolved_Oxygen","Temperature","Conductivity","BOD","COD"]

print("=" * 70)
print("  STEP 1: DATA QUALITY CHECK")
print("=" * 70)

df = load_data("data/water_quality.csv")
print(f"\n  Shape: {df.shape}")
print(f"\n  Missing values:")
for col in features:
    missing = df[col].isna().sum() if col in df.columns else "COLUMN MISSING"
    print(f"    {col}: {missing}")

print(f"\n  Value ranges:")
for col in features:
    if col in df.columns:
        print(f"    {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}, std={df[col].std():.2f}")

print(f"\n  Outliers (beyond 3 std):")
for col in features:
    if col in df.columns:
        mean, std = df[col].mean(), df[col].std()
        outliers = ((df[col] < mean - 3*std) | (df[col] > mean + 3*std)).sum()
        print(f"    {col}: {outliers} outliers")

print(f"\n  Autocorrelation (lag-1):")
for col in features:
    if col in df.columns:
        corr = df[col].autocorr(lag=1)
        print(f"    {col}: {corr:.4f}")

print("\n" + "=" * 70)
print("  STEP 2: BASELINE MODELS")
print("=" * 70)

X_train, X_val, X_test, y_train, y_val, y_test, f_scaler, t_scaler = preprocess_data(df)
X_train_flat = X_train[:, -1, :]
X_test_flat = X_test[:, -1, :]
y_train_orig = t_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_orig = t_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

lr = LinearRegression()
lr.fit(X_train_flat, y_train_orig)
lr_pred = lr.predict(X_test_flat)
lr_rmse = np.sqrt(mean_squared_error(y_test_orig, lr_pred))
lr_r2 = r2_score(y_test_orig, lr_pred)
print(f"\n  Linear Regression: RMSE={lr_rmse:.4f}, R2={lr_r2:.4f}")

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_flat, y_train_orig)
rf_pred = rf.predict(X_test_flat)
rf_rmse = np.sqrt(mean_squared_error(y_test_orig, rf_pred))
rf_r2 = r2_score(y_test_orig, rf_pred)
print(f"  Random Forest: RMSE={rf_rmse:.4f}, R2={rf_r2:.4f}")

print(f"\n  RF Feature Importance:")
for name, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"    {name}: {imp:.4f}")

print("\n" + "=" * 70)
print("  STEP 3: TARGET ALIGNMENT CHECK")
print("=" * 70)
print(f"  y_train: mean={y_train_orig.mean():.2f}, std={y_train_orig.std():.2f}")
print(f"  y_test:  mean={y_test_orig.mean():.2f}, std={y_test_orig.std():.2f}")

print(f"\n  Correlation last-step features vs T+1 target:")
for i, name in enumerate(features):
    corr = np.corrcoef(X_train_flat[:, i], y_train_orig)[0, 1]
    print(f"    {name} vs WQI(T+1): {corr:.4f}")

print("\n" + "=" * 70)
print("  STEP 4: ANALYSIS & RECOMMENDATIONS")
print("=" * 70)
print(f"  Features: {features}")
print(f"  Expected LSTM R2: {max(0.4, rf_r2-0.1):.2f} to {min(0.95, rf_r2+0.1):.2f}")
print("=" * 70)
