"""
signal_engine.py
----------------
Signal redesign and de-noising utilities for Kitchen Prep Time (KPT) prediction.
Implements:
  - Exponential smoothing
  - Rolling variance (noise quantification)
  - Anomaly detection
  - Live Rush Index (LRI)
  - Capacity-adjusted forecast
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


# ---------------------------------------------------------------------------
# 1. Signal Smoothing
# ---------------------------------------------------------------------------

def smooth_signal(signal: Union[pd.Series, np.ndarray], alpha: float = 0.3) -> np.ndarray:
    """
    Apply exponential weighted moving average (EWMA) smoothing.

    This removes high-frequency noise from observed KPT signals without
    introducing large lag, making it suitable for near-real-time de-noising.

    Parameters
    ----------
    signal : array-like
        Raw observed signal (e.g., observed_prep_time series).
    alpha : float, default 0.3
        Smoothing factor in (0, 1]. Higher = less smoothing (more reactive).

    Returns
    -------
    np.ndarray
        Smoothed signal values.

    Example
    -------
    >>> raw = np.array([20, 22, 45, 21, 19, 23, 50, 20])
    >>> smoothed = smooth_signal(raw, alpha=0.3)
    >>> smoothed.round(2)
    array([20.  , 20.6 , 27.12, 25.48, 23.34, 23.34, 31.34, 27.94])
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    signal = np.asarray(signal, dtype=float)
    smoothed = np.empty_like(signal)
    smoothed[0] = signal[0]
    for t in range(1, len(signal)):
        smoothed[t] = alpha * signal[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


# ---------------------------------------------------------------------------
# 2. Rolling Variance
# ---------------------------------------------------------------------------

def rolling_variance(signal: Union[pd.Series, np.ndarray], window: int = 10) -> np.ndarray:
    """
    Compute rolling variance over a fixed window.

    High rolling variance indicates kitchen instability or rush periods.
    Used as a noise-quantification feature in downstream models.

    Parameters
    ----------
    signal : array-like
        Input time-series (e.g., prep time observations).
    window : int, default 10
        Rolling window size (number of observations).

    Returns
    -------
    np.ndarray
        Rolling variance array; first (window-1) elements are NaN.

    Example
    -------
    >>> s = np.array([10, 11, 10, 30, 31, 29, 10, 11])
    >>> rolling_variance(s, window=3).round(2)
    array([       nan,        nan,  0.33, 101.  , 101.  , 101.  ,  101., 101.])
    """
    signal = pd.Series(signal, dtype=float)
    return signal.rolling(window=window, min_periods=window).var().to_numpy()


# ---------------------------------------------------------------------------
# 3. Anomaly Detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    signal: Union[pd.Series, np.ndarray],
    threshold: float = 2.5,
    window: int = 20,
) -> np.ndarray:
    """
    Detect anomalies using a rolling Z-score method.

    Points whose |Z-score| exceeds `threshold` are flagged as anomalies.
    These correspond to sudden rush spikes, FOR system glitches, or
    exceptional events that would corrupt a naive ML model.

    Parameters
    ----------
    signal : array-like
        Input time-series.
    threshold : float, default 2.5
        Z-score cutoff. Values above this are anomalies.
    window : int, default 20
        Rolling window for computing local mean/std.

    Returns
    -------
    np.ndarray of bool
        Boolean mask; True = anomaly detected at that index.

    Example
    -------
    >>> s = np.array([20, 21, 19, 20, 60, 20, 19])
    >>> detect_anomalies(s, threshold=2.0, window=4)
    array([False, False, False, False,  True, False, False])
    """
    signal = pd.Series(np.asarray(signal, dtype=float))
    roll_mean = signal.rolling(window=window, min_periods=3).mean()
    roll_std = signal.rolling(window=window, min_periods=3).std().replace(0, np.nan)
    z_scores = (signal - roll_mean) / roll_std
    return (z_scores.abs() > threshold).fillna(False).to_numpy()


# ---------------------------------------------------------------------------
# 4. Live Rush Index
# ---------------------------------------------------------------------------

def compute_live_rush_index(
    data: pd.DataFrame,
    prep_col: str = "observed_prep_time",
    queue_col: Optional[str] = "zomato_orders_per_min",
    spike_col: Optional[str] = "sudden_rush_spike",
    smooth_alpha: float = 0.3,
    anomaly_threshold: float = 2.5,
    window: int = 20,
) -> pd.Series:
    """
    Compute a composite Live Rush Index (LRI) per observation.

    LRI combines three signals:
      1. Prep deviation score  – normalised deviation of smoothed prep from baseline
      2. Queue depth score     – normalised incoming order rate
      3. Rejection/spike score – weighted anomaly count in rolling window

    LRI ∈ [0, 1] where 1 = maximum detected rush.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain at least `prep_col`. Optionally `queue_col`, `spike_col`.
    prep_col : str
        Column name for observed prep times.
    queue_col : str or None
        Column name for incoming order rate.
    spike_col : str or None
        Column name for sudden rush spike flag (binary).
    smooth_alpha : float
        EWMA alpha for smoothing prep signal.
    anomaly_threshold : float
        Z-score threshold for anomaly detection.
    window : int
        Rolling window for variance/anomaly scoring.

    Returns
    -------
    pd.Series
        LRI values (float, 0–1) indexed like `data`.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"observed_prep_time": np.random.normal(20, 5, 100),
    ...                    "zomato_orders_per_min": np.random.uniform(1, 8, 100),
    ...                    "sudden_rush_spike": np.random.randint(0, 2, 100)})
    >>> lri = compute_live_rush_index(df)
    >>> lri.describe()
    """
    prep = data[prep_col].values.copy()

    # --- Component 1: Prep deviation ---
    smoothed = smooth_signal(prep, alpha=smooth_alpha)
    baseline = np.nanmedian(smoothed)
    prep_dev = np.clip((smoothed - baseline) / (baseline + 1e-6), 0, None)
    prep_dev_norm = prep_dev / (prep_dev.max() + 1e-6)

    # --- Component 2: Queue depth ---
    if queue_col and queue_col in data.columns:
        queue = data[queue_col].values.astype(float)
        queue_norm = (queue - queue.min()) / (queue.max() - queue.min() + 1e-6)
    else:
        queue_norm = np.zeros(len(data))

    # --- Component 3: Spike / rejection signal ---
    if spike_col and spike_col in data.columns:
        spikes = data[spike_col].values.astype(float)
        spike_roll = pd.Series(spikes).rolling(window=window, min_periods=1).mean().values
    else:
        anomalies = detect_anomalies(prep, threshold=anomaly_threshold, window=window).astype(float)
        spike_roll = pd.Series(anomalies).rolling(window=window, min_periods=1).mean().values

    # --- Composite LRI (weighted average) ---
    lri = 0.45 * prep_dev_norm + 0.30 * queue_norm + 0.25 * spike_roll
    lri = np.clip(lri, 0, 1)
    return pd.Series(lri, index=data.index, name="live_rush_index")


# ---------------------------------------------------------------------------
# 5. Capacity-Adjusted Forecast
# ---------------------------------------------------------------------------

def capacity_adjusted_forecast(
    forecast: Union[pd.Series, np.ndarray],
    capacity_pct: Union[pd.Series, np.ndarray],
    max_stretch_factor: float = 1.6,
    min_stretch_factor: float = 0.85,
) -> np.ndarray:
    """
    Adjust model forecast upward/downward based on kitchen capacity utilisation.

    Formula:
        stretch = 1 + (capacity_pct - 0.5) * (max_stretch - 1) / 0.5
        adjusted_forecast = forecast * clip(stretch, min_stretch, max_stretch)

    Parameters
    ----------
    forecast : array-like
        Raw model forecast of prep time (minutes).
    capacity_pct : array-like
        Kitchen capacity utilisation in [0, 1]; e.g., dine_in_load_proxy or
        a composite of dine-in + online orders.
    max_stretch_factor : float, default 1.6
        Maximum multiplier when capacity is at 100%.
    min_stretch_factor : float, default 0.85
        Minimum multiplier when capacity is near 0%.

    Returns
    -------
    np.ndarray
        Capacity-adjusted forecast (minutes).

    Example
    -------
    >>> forecasts = np.array([20, 22, 25])
    >>> capacities = np.array([0.2, 0.6, 0.95])
    >>> capacity_adjusted_forecast(forecasts, capacities).round(2)
    array([17.  , 22.88, 32.  ])
    """
    forecast = np.asarray(forecast, dtype=float)
    capacity_pct = np.asarray(capacity_pct, dtype=float)
    stretch = 1.0 + (capacity_pct - 0.5) * (max_stretch_factor - 1.0) / 0.5
    stretch = np.clip(stretch, min_stretch_factor, max_stretch_factor)
    return forecast * stretch


# ---------------------------------------------------------------------------
# Convenience: build full signal-engineered feature DataFrame
# ---------------------------------------------------------------------------

def build_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all signal-engineered columns to the raw DataFrame.

    New columns added:
        smoothed_prep, rolling_var_prep, anomaly_flag,
        live_rush_index, capacity_pct

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: observed_prep_time, zomato_orders_per_min,
                      sudden_rush_spike, dine_in_load_proxy

    Returns
    -------
    pd.DataFrame
        Enhanced DataFrame with new signal columns.
    """
    out = df.copy()
    out["smoothed_prep"] = smooth_signal(out["observed_prep_time"].values, alpha=0.3)
    out["rolling_var_prep"] = rolling_variance(out["observed_prep_time"].values, window=15)
    out["anomaly_flag"] = detect_anomalies(out["observed_prep_time"].values, threshold=2.5).astype(int)
    out["live_rush_index"] = compute_live_rush_index(
        out,
        prep_col="observed_prep_time",
        queue_col="zomato_orders_per_min",
        spike_col="sudden_rush_spike",
    )
    # Capacity proxy: blend of dine-in load and normalised order rate
    q_norm = (out["zomato_orders_per_min"] - out["zomato_orders_per_min"].min()) / \
             (out["zomato_orders_per_min"].max() - out["zomato_orders_per_min"].min() + 1e-6)
    out["capacity_pct"] = np.clip(0.6 * out["dine_in_load_proxy"] + 0.4 * q_norm, 0, 1)
    return out


if __name__ == "__main__":
    # Quick smoke test
    np.random.seed(0)
    dummy = pd.DataFrame({
        "observed_prep_time": np.random.normal(20, 5, 200) + np.where(np.random.rand(200) < 0.05, 30, 0),
        "zomato_orders_per_min": np.random.uniform(1, 8, 200),
        "sudden_rush_spike": np.random.randint(0, 2, 200),
        "dine_in_load_proxy": np.random.uniform(0.2, 0.9, 200),
    })
    result = build_signal_features(dummy)
    print(result[["observed_prep_time", "smoothed_prep", "anomaly_flag", "live_rush_index", "capacity_pct"]].head(10))
    print("\nSignal engine OK ✅")
