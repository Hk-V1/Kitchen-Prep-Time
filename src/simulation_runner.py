"""
simulation_runner.py
--------------------
Runs three simulation scenarios and compares KPT prediction quality:

  1. Baseline  – raw noisy FOR signals, simple mean predictor
  2. ML-only   – XGBoost trained on raw (un-engineered) signals
  3. ASIF      – Signal-engineered + Live Rush Index + Capacity Adjustment

Outputs:
  - Console table
  - /data/simulation_results.csv
  - /data/plots/simulation_comparison.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Allow imports from /src
sys.path.insert(0, os.path.dirname(__file__))

from signal_engine import build_signal_features, capacity_adjusted_forecast
from forecast_model import train_and_evaluate, apply_capacity_adjustment
from metrics import compute_all_metrics, print_metrics_table

sns.set_theme(style="whitegrid", palette="muted")

PLOT_DIR = "../data/plots"
RESULTS_CSV = "../data/simulation_results.csv"
DATA_PATH = "../data/synthetic_orders.csv"


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def scenario_baseline(df: pd.DataFrame) -> dict:
    """
    Baseline: predict using rolling mean of RAW observed_prep_time.
    Represents the current FOR system with no ML or signal engineering.
    """
    print("\n[Scenario 1] Running BASELINE (rolling mean of noisy FOR signal)...")
    # Rolling mean of last 20 observations per restaurant as the "prediction"
    df_sorted = df.sort_values("timestamp").copy()
    df_sorted["baseline_pred"] = (
        df_sorted.groupby("restaurant_id")["observed_prep_time"]
        .transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    )
    df_sorted["baseline_pred"] = df_sorted["baseline_pred"].fillna(df_sorted["observed_prep_time"].mean())

    y_true = df_sorted["true_prep_time"].values
    y_pred = df_sorted["baseline_pred"].values
    return compute_all_metrics(y_true, y_pred, label="1. Baseline (noisy FOR)")


def scenario_ml_only(df: pd.DataFrame) -> dict:
    """
    ML-only: XGBoost trained on raw signals (no signal engineering).
    """
    print("\n[Scenario 2] Running ML-ONLY (XGBoost on raw features)...")
    result = train_and_evaluate(df, use_engineered=False, label="ML-only")
    return compute_all_metrics(result["y_test"], result["y_pred"], label="2. ML-Only (raw features)")


def scenario_asif(df_engineered: pd.DataFrame) -> dict:
    """
    ASIF: Signal-engineered features + Live Rush Index + Capacity Adjustment.
    """
    print("\n[Scenario 3] Running ASIF (Signal Engineering + LRI + Capacity Adj)...")
    result = train_and_evaluate(df_engineered, use_engineered=True, label="ASIF")

    # Apply capacity adjustment on test set
    cap_pct = np.full(len(result["y_pred"]), df_engineered["capacity_pct"].mean())
    y_pred_adj = apply_capacity_adjustment(result["y_pred"], cap_pct)

    return compute_all_metrics(result["y_test"], y_pred_adj, label="3. ASIF System")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(results_df: pd.DataFrame, output_dir: str = PLOT_DIR):
    """Generate bar-chart comparison of all metrics across scenarios."""
    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = {
        "MAE (min)": "MAE",
        "RMSE (min)": "RMSE",
        "SLA % ↑": "SLA_%",
        "Rider Idle (min)": "Rider_Idle_min",
        "Cancellation % ↓": "Cancellation_%",
    }

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(22, 6))
    fig.suptitle("KPT Prediction: Scenario Comparison", fontsize=16, fontweight="bold", y=1.02)

    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    scenarios = results_df["label"].tolist()

    for ax, (title, col) in zip(axes, metrics_to_plot.items()):
        vals = results_df[col].values
        bars = ax.bar(range(len(scenarios)), vals, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(["Baseline", "ML-Only", "ASIF"], fontsize=9, rotation=15, ha="right")

        # Annotate bars
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(vals),
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    path = os.path.join(output_dir, "simulation_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Plot saved → {path}")


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, label: str, output_dir: str = PLOT_DIR):
    """Scatter plot of predicted vs actual for a given scenario."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true[:500], y_pred[:500], alpha=0.3, s=15, color="#3498db")
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual KPT (min)", fontsize=12)
    ax.set_ylabel("Predicted KPT (min)", fontsize=12)
    ax.set_title(f"Predicted vs Actual – {label}", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("/", "_")
    path = os.path.join(output_dir, f"pred_vs_actual_{safe_label}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   Scatter plot → {path}")


def plot_signal_denoising(df: pd.DataFrame, restaurant_id: int = 1, output_dir: str = PLOT_DIR):
    """Show raw vs smoothed signal for a single restaurant."""
    os.makedirs(output_dir, exist_ok=True)
    sub = df[df["restaurant_id"] == restaurant_id].sort_values("timestamp").head(200)
    if sub.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Raw vs smoothed
    axes[0].plot(sub.index, sub["observed_prep_time"], alpha=0.5, color="#e74c3c", label="Raw observed", linewidth=1)
    if "smoothed_prep" in sub.columns:
        axes[0].plot(sub.index, sub["smoothed_prep"], color="#2c3e50", label="Smoothed", linewidth=1.8)
    axes[0].axhline(sub["true_prep_time"].mean(), color="green", linestyle="--", alpha=0.7, label="True mean")
    axes[0].set_ylabel("Prep Time (min)", fontsize=11)
    axes[0].set_title(f"Signal De-noising – Restaurant {restaurant_id}", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)

    # Live Rush Index
    if "live_rush_index" in sub.columns:
        axes[1].fill_between(sub.index, sub["live_rush_index"], alpha=0.4, color="#e67e22", label="Live Rush Index")
        axes[1].plot(sub.index, sub["live_rush_index"], color="#e67e22", linewidth=1)
        axes[1].set_ylabel("Live Rush Index", fontsize=11)
        axes[1].set_xlabel("Observation index", fontsize=11)
        axes[1].set_title("Live Rush Index (LRI)", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "signal_denoising.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   Signal plot → {path}")


# Main

def run_simulation():
    """End-to-end simulation comparing all three scenarios."""
    print("\n" + "=" * 65)
    print("   KPT PREDICTION SIMULATION – ASIF vs BASELINE vs ML-ONLY")
    print("=" * 65)

    # Load or generate data
    if not os.path.exists(DATA_PATH):
        print("Generating synthetic data first...")
        from synthetic_data import generate_all_restaurants
        df_raw = generate_all_restaurants(30, 30, DATA_PATH)
    else:
        df_raw = pd.read_csv(DATA_PATH)
        print(f" Loaded data: {df_raw.shape} rows")

    # Build signal-engineered features
    print("\nApplying signal engineering...")
    df_eng = build_signal_features(df_raw)
    print(f"   New columns: {[c for c in df_eng.columns if c not in df_raw.columns]}")

    # Plot signal de-noising
    plot_signal_denoising(df_eng, restaurant_id=1)

    # Run three scenarios
    r_baseline = scenario_baseline(df_raw)
    r_ml = scenario_ml_only(df_raw)
    r_asif = scenario_asif(df_eng)

    # Collect and display
    all_results = [r_baseline, r_ml, r_asif]
    results_df = print_metrics_table(all_results)

    # Save CSV
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    results_df.reset_index().to_csv(RESULTS_CSV, index=False)
    print(f"\n Results saved → {RESULTS_CSV}")

    # Print improvement summary
    baseline_mae = r_baseline["MAE"]
    asif_mae = r_asif["MAE"]
    mae_improvement = (baseline_mae - asif_mae) / baseline_mae * 100

    baseline_sla = r_baseline["SLA_%"]
    asif_sla = r_asif["SLA_%"]

    print(f"\n KEY IMPROVEMENTS (Baseline → ASIF):")
    print(f"   MAE   : {baseline_mae:.2f} → {asif_mae:.2f} min  ({mae_improvement:.1f}% ↓)")
    print(f"   SLA % : {baseline_sla:.1f}% → {asif_sla:.1f}%  (+{asif_sla - baseline_sla:.1f}pp)")
    print(f"   Rider Idle: {r_baseline['Rider_Idle_min']:.2f} → {r_asif['Rider_Idle_min']:.2f} min/order")
    print(f"   Cancel Rate: {r_baseline['Cancellation_%']:.1f}% → {r_asif['Cancellation_%']:.1f}%")

    # Plots
    plot_comparison(results_df.reset_index())

    return results_df


if __name__ == "__main__":
    run_simulation()
