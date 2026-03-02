"""
forecast_model.py
-----------------
Multi-signal KPT forecast using XGBoost (falls back to GradientBoosting if
xgboost is not installed). Takes both raw and signal-engineered features.
"""

import sys
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    _USE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _USE_XGB = False
    print("[WARN] xgboost not found – falling back to sklearn GradientBoostingRegressor")


# Feature definitions

RAW_FEATURES = [
    "zomato_orders_per_min",
    "promo_flag",
    "weather_effect",
    "local_event_flag",
    "dine_in_load_proxy",
    "sudden_rush_spike",
    "observed_prep_time",
    "hour",
    "day_of_week",
    "is_weekend",
]

ENGINEERED_FEATURES = RAW_FEATURES + [
    "smoothed_prep",
    "rolling_var_prep",
    "anomaly_flag",
    "live_rush_index",
    "capacity_pct",
]

TARGET = "true_prep_time"

# Feature engineering helpers

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day_of_week, is_weekend from timestamp column."""
    df = df.copy()
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def prepare_dataset(df: pd.DataFrame, use_engineered: bool = True):
    """
    Split dataset into X / y with appropriate feature set.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (output of generate_synthetic_data + signal_engine).
    use_engineered : bool
        If True, use signal-engineered features; else raw-only.

    Returns
    -------
    X_train, X_test, y_train, y_test : split arrays
    feature_cols : list of str
    """
    df = add_time_features(df)
    feature_cols = ENGINEERED_FEATURES if use_engineered else RAW_FEATURES
    available = [c for c in feature_cols if c in df.columns]
    df_clean = df.dropna(subset=available + [TARGET])

    X = df_clean[available].values
    y = df_clean[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    return X_train, X_test, y_train, y_test, available

# Model builder

def build_model(use_engineered: bool = True):
    """Return a fresh regressor instance."""
    if _USE_XGB:
        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

# Train & evaluate

def train_and_evaluate(
    df: pd.DataFrame,
    use_engineered: bool = True,
    label: str = "Model",
) -> dict:
    """
    Train a forecast model and return performance metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features and target.
    use_engineered : bool
        Whether to use signal-engineered features.
    label : str
        Label for logging.

    Returns
    -------
    dict with keys: label, mae, rmse, model, feature_cols,
                    y_test, y_pred, X_test, scaler
    """
    X_train, X_test, y_train, y_test, feature_cols = prepare_dataset(df, use_engineered)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = build_model(use_engineered)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_pred = np.clip(y_pred, 1, None)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    print(f"\n{'='*50}")
    print(f" {label} ({'Engineered' if use_engineered else 'Raw'} features)")
    print(f"{'='*50}")
    print(f"  MAE  : {mae:.3f} min")
    print(f"  RMSE : {rmse:.3f} min")
    print(f"  Features used ({len(feature_cols)}): {feature_cols}")

    return {
        "label": label,
        "mae": mae,
        "rmse": rmse,
        "model": model,
        "feature_cols": feature_cols,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test": X_test_sc,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Capacity-adjusted final prediction
# ---------------------------------------------------------------------------

def apply_capacity_adjustment(y_pred: np.ndarray, capacity_pct: np.ndarray) -> np.ndarray:
    """Wrap capacity_adjusted_forecast from signal_engine."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from signal_engine import capacity_adjusted_forecast
    return capacity_adjusted_forecast(y_pred, capacity_pct)


# ---------------------------------------------------------------------------
# Main: run both raw-only and engineered models for comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from synthetic_data import generate_all_restaurants
    from signal_engine import build_signal_features

    DATA_PATH = "../data/synthetic_orders.csv"
    if not os.path.exists(DATA_PATH):
        df_raw = generate_all_restaurants(30, 30, DATA_PATH)
    else:
        df_raw = pd.read_csv(DATA_PATH)

    df = build_signal_features(df_raw)

    res_raw = train_and_evaluate(df, use_engineered=False, label="Raw-Only Model")
    res_eng = train_and_evaluate(df, use_engineered=True, label="ASIF Engineered Model")

    print("\n\n📊 Comparison Summary:")
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8}")
    print("-" * 48)
    for r in [res_raw, res_eng]:
        print(f"{r['label']:<30} {r['mae']:>8.3f} {r['rmse']:>8.3f}")
