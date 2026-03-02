"""
metrics.py
----------
Evaluation metrics for KPT prediction system.
Covers: MAE, RMSE, SLA %, Rider Idle Proxy, Cancellation Proxy.
"""

import numpy as np
import pandas as pd
from typing import Union


ArrayLike = Union[np.ndarray, pd.Series, list]


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Absolute Error.

    Parameters
    ----------
    y_true : array-like
        Ground truth prep times (minutes).
    y_pred : array-like
        Predicted prep times (minutes).

    Returns
    -------
    float : MAE in minutes.

    Example
    -------
    >>> mae([20, 25, 30], [18, 27, 28])
    2.0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error.

    Penalises large deviations more than MAE, making it sensitive to rush spikes.

    Parameters
    ----------
    y_true : array-like
        Ground truth prep times.
    y_pred : array-like
        Predicted prep times.

    Returns
    -------
    float : RMSE in minutes.

    Example
    -------
    >>> rmse([20, 25, 30], [18, 27, 28])
    2.160...
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def sla_percentage(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    tolerance_min: float = 5.0,
) -> float:
    """
    SLA %: Fraction of orders where |predicted - actual| ≤ tolerance.

    An order is "SLA-compliant" when the KPT prediction is accurate enough
    that the rider arrives within the acceptable window (+/- tolerance).

    Parameters
    ----------
    y_true : array-like
        Actual prep times.
    y_pred : array-like
        Predicted prep times.
    tolerance_min : float, default 5.0
        Acceptable prediction error in minutes.

    Returns
    -------
    float : SLA percentage in [0, 100].

    Example
    -------
    >>> sla_percentage([20, 25, 30, 15], [19, 31, 28, 14], tolerance_min=5)
    75.0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    within_sla = np.abs(y_true - y_pred) <= tolerance_min
    return float(100 * within_sla.mean())


def rider_idle_proxy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    arrival_buffer_min: float = 2.0,
) -> float:
    """
    Rider Idle Time Proxy (minutes per order).

    When prediction underestimates prep time, the rider arrives early and waits.
    This models the average excess wait time across all under-predicted orders.

    idle_time = max(0, y_true - y_pred - arrival_buffer)

    Parameters
    ----------
    y_true : array-like
        Actual prep times.
    y_pred : array-like
        Predicted prep times.
    arrival_buffer_min : float, default 2.0
        Grace period (minutes) built into dispatch logic.

    Returns
    -------
    float : Average rider idle time per order (minutes).

    Example
    -------
    >>> rider_idle_proxy([25, 20, 30], [18, 22, 28], arrival_buffer_min=2)
    1.667...
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    idle = np.maximum(0, (y_true - y_pred) - arrival_buffer_min)
    return float(idle.mean())


def cancellation_proxy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    overestimate_threshold: float = 8.0,
) -> float:
    """
    Cancellation Rate Proxy (%).

    When predicted KPT is much higher than actual, platform may show inflated
    ETAs, causing customer cancellations. This proxy estimates that rate.

    A "likely cancellation" occurs when:
        y_pred - y_true > overestimate_threshold

    Parameters
    ----------
    y_true : array-like
        Actual prep times.
    y_pred : array-like
        Predicted prep times.
    overestimate_threshold : float, default 8.0
        Minutes of over-prediction that would trigger a cancellation.

    Returns
    -------
    float : Estimated cancellation rate (%).

    Example
    -------
    >>> cancellation_proxy([20, 20, 20], [20, 25, 32], overestimate_threshold=8)
    33.333...
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    over_predicted = (y_pred - y_true) > overestimate_threshold
    return float(100 * over_predicted.mean())


def compute_all_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    tolerance_min: float = 5.0,
    arrival_buffer_min: float = 2.0,
    overestimate_threshold: float = 8.0,
    label: str = "Model",
) -> dict:
    """
    Compute all KPT metrics and return as a dictionary.

    Parameters
    ----------
    y_true : array-like
        Ground truth prep times.
    y_pred : array-like
        Predicted prep times.
    tolerance_min : float
        SLA tolerance (minutes).
    arrival_buffer_min : float
        Rider idle grace period (minutes).
    overestimate_threshold : float
        Cancellation trigger threshold (minutes of over-prediction).
    label : str
        Model/scenario label for display.

    Returns
    -------
    dict with keys: label, MAE, RMSE, SLA_pct, Rider_Idle, Cancellation_Rate
    """
    return {
        "label": label,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "SLA_%": sla_percentage(y_true, y_pred, tolerance_min),
        "Rider_Idle_min": rider_idle_proxy(y_true, y_pred, arrival_buffer_min),
        "Cancellation_%": cancellation_proxy(y_true, y_pred, overestimate_threshold),
    }


def print_metrics_table(metrics_list: list) -> pd.DataFrame:
    """
    Pretty-print a comparison table of metrics across scenarios.

    Parameters
    ----------
    metrics_list : list of dict
        Each dict is output of compute_all_metrics().

    Returns
    -------
    pd.DataFrame
        Formatted comparison table.
    """
    df = pd.DataFrame(metrics_list).set_index("label")
    df = df.round(3)
    print("\n" + "=" * 70)
    print("  KPT PREDICTION SYSTEM — METRICS COMPARISON")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)
    return df


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    y_true = np.random.normal(22, 4, 500)

    # Baseline: noisy
    y_baseline = y_true + np.random.normal(0, 7, 500)
    # ML only: moderate improvement
    y_ml = y_true + np.random.normal(0, 4, 500)
    # ASIF: best
    y_asif = y_true + np.random.normal(0, 2.2, 500)

    results = [
        compute_all_metrics(y_true, y_baseline, label="Baseline (noisy FOR)"),
        compute_all_metrics(y_true, y_ml, label="ML-only"),
        compute_all_metrics(y_true, y_asif, label="ASIF System"),
    ]
    print_metrics_table(results)
