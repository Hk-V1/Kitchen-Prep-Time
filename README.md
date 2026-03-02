# 🍽️ KPT Prediction — ASIF + KSAL Combined Solution

### Hackathon Submission: Kitchen Prep Time Prediction via Signal Redesign, De-noising, Live Rush Detection & Capacity-Aware Forecasting

---

## Problem Statement

Food delivery platforms lose revenue and customer trust because Kitchen Prep Time (KPT) forecasts are inaccurate. The root cause is not a weak ML model — it is **corrupted input signals**:

- Merchant-marked FOR (Food Order Ready) signals are noisy and biased
- No real-time visibility into kitchen load (dine-in, concurrent batches, competitor orders)
- Models ignore external context (promos, weather, local events)
- No capacity-based adjustment to final forecasts

**Result:** Early rider dispatch → idle wait → bad ETA → customer cancellations.

---

## Solution: KSAL + ASIF

This submission combines two complementary layers into one unified pipeline:

| Layer | Full Name | What It Does |
|-------|-----------|--------------|
| **KSAL** | Kitchen Signal Augmentation Layer | Replaces noisy FOR labels with real instrumented cooking signals |
| **ASIF** | Adaptive Signal-Integrated Forecast | De-noises signals, computes Live Rush Index, trains ML, adjusts for capacity |

```
[Raw FOR / Stove Sensors / POS] 
        ↓ KSAL: RCST + HRF + SUF + RTRM
[Enriched Signals + External Context]
        ↓ ASIF: EWMA + Anomaly Detection + LRI
[Engineered Features]
        ↓ ML Forecast (XGBoost / GradientBoosting)
        ↓ Capacity Adjustment
[Adjusted KPT Prediction → Rider Dispatch]
        ↓
[SLA ↑ · Rider Idle ↓ · Cancellations ↓]
```

---

## Folder Structure

```
kpt_hackathon/
├── data/
│   ├── synthetic_orders.csv        # 218K+ orders, 30 restaurants × 30 days
│   ├── simulation_results.csv      # 3-scenario comparison output
│   └── plots/
│       ├── signal_denoising.png    # Raw vs smoothed + LRI chart
│       ├── signal_analysis.png     # Variance + anomaly overlay
│       ├── feature_importance.png  # ASIF model feature ranking
│       ├── pred_vs_actual.png      # Predicted vs actual scatter
│       ├── simulation_comparison.png  # 5-metric bar chart
│       └── final_comparison.png    # Full metrics comparison
├── src/
│   ├── generate_synthetic_data.py  # Synthetic dataset generation
│   ├── signal_engine.py            # EWMA, variance, anomaly, LRI, capacity adj.
│   ├── forecast_model.py           # XGBoost / GradientBoosting model
│   ├── simulation_runner.py        # 3-scenario simulation + plots
│   ├── metrics.py                  # MAE, RMSE, SLA%, idle, cancellation
│   └── generate_architecture_diagram.py  # Saves diagrams/asif_architecture.png
├── notebooks/
│   └── KPT_ASIF_Demo.ipynb         # Full step-by-step demo with inline plots
├── diagrams/
│   └── asif_architecture.png       # System architecture diagram
└── README.md
```

---

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost
```

> XGBoost is optional — falls back to `sklearn.GradientBoostingRegressor` automatically if not installed.

---

## How to Run

### Step 1: Generate Synthetic Data
```bash
cd src
python generate_synthetic_data.py
# Output: ../data/synthetic_orders.csv (218K+ rows)
```

### Step 2: Inspect Signal Engine
```bash
python signal_engine.py
# Runs smoke test: smoothing, variance, anomaly detection, LRI, capacity adjustment
```

### Step 3: Train Forecast Models
```bash
python forecast_model.py
# Trains both raw-only and ASIF-engineered models
# Prints MAE / RMSE comparison table
```

### Step 4: Full Simulation
```bash
python simulation_runner.py
# Runs 3 scenarios: Baseline / ML-Only / ASIF
# Saves plots to ../data/plots/
# Saves results to ../data/simulation_results.csv
```

### Step 5: Notebook Demo
```bash
cd ../notebooks
jupyter notebook KPT_ASIF_Demo.ipynb
# Full walkthrough with inline visualisations
```

### Step 6: Architecture Diagram
```bash
cd ../src
python generate_architecture_diagram.py
# Output: ../diagrams/asif_architecture.png
```

---

## KSAL — Signal Augmentation Details

KSAL replaces noisy merchant-marked FOR timestamps with four real signals:

| Signal | Formula | What It Captures |
|--------|---------|-----------------|
| **RCST** | Stove activation / POS timestamp | Actual cooking start (not order confirmation) |
| **HRF** | Avg actual cook time ÷ base recipe time | Predictable peak-hour slowdown |
| **SUF** | Active stoves ÷ total stoves | Real-time kitchen capacity pressure |
| **RTRM** | HRF × (1 + SUF) | Combined dynamic prep multiplier |

**Deployment tiers:**
- **Tier 1 (large chains):** Smart plug + full sensor stack
- **Tier 2 (mid-size):** POS timestamps + historical HRF
- **Tier 3 (small restaurants):** Software-only historical multipliers

---

## 🔬 ASIF — Signal Engineering Details

### `signal_engine.py`

| Function | Description |
|----------|-------------|
| `smooth_signal(signal, alpha)` | EWMA de-noising — lower lag than simple MA, better for real-time dispatch |
| `rolling_variance(signal, window)` | Kitchen instability quantification over rolling window |
| `detect_anomalies(signal, threshold)` | Rolling Z-score flags rush spikes before they corrupt training labels |
| `compute_live_rush_index(data)` | Composite LRI [0–1]: 45% prep deviation + 30% queue depth + 25% spike rate |
| `capacity_adjusted_forecast(forecast, pct)` | Post-model scaling: stretches forecast up (high load) or compresses (low load) |

### `metrics.py`

| Function | What It Measures |
|----------|----------------|
| `mae(y_true, y_pred)` | Mean Absolute Error (minutes) |
| `rmse(y_true, y_pred)` | Root Mean Squared Error — penalises large rush-period errors |
| `sla_percentage(...)` | % orders predicted within ±5 min |
| `rider_idle_proxy(...)` | Avg minutes rider waits at restaurant |
| `cancellation_proxy(...)` | % orders where over-prediction exceeds 8 min (cancellation risk) |

---

## 📊 Results

| Scenario | MAE (min) | RMSE (min) | SLA % | Rider Idle | Cancel Rate |
|----------|-----------|------------|-------|------------|-------------|
| Baseline (noisy FOR) | ~7.0 | ~9.5 | ~58% | ~4.5 min | ~22% |
| ML-Only (raw features) | ~4.5 | ~6.1 | ~71% | ~2.8 min | ~14% |
| **ASIF System** | **~2.2** | **~3.1** | **~87%** | **~1.1 min** | **~6%** |

**Improvements over baseline:**
- ✅ **MAE** reduced by ~68%
- ✅ **SLA compliance** up by ~29 percentage points
- ✅ **Rider idle time** cut by ~76%
- ✅ **Cancellation rate** reduced by ~73%

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| EWMA over simple MA | Lower dispatch lag at rush onset — critical for real-time use |
| LRI as ML feature (not just filter) | Gives model direct awareness of kitchen stress; more interpretable |
| Anomaly detection *before* training | Removes rush spike contamination from training labels at source |
| Capacity adjustment post-ML | Decouples operational state from model; easier to tune and explain |
| KSAL tiered deployment | Hardware cost scales with restaurant size; software fallback for 300K+ merchants |
| Multi-source external signals | Weather + promo + local events as first-class features — proven MAE improvement |

---

## Visualisations

After `python simulation_runner.py`, find charts in `/data/plots/`:

- **`signal_denoising.png`** — Raw vs EWMA smoothed + Live Rush Index over time
- **`signal_analysis.png`** — Rolling variance + anomaly overlay
- **`feature_importance.png`** — ASIF model feature importance ranking
- **`pred_vs_actual.png`** — Scatter plot: predicted vs actual KPT
- **`simulation_comparison.png`** — 5-metric bar chart across all 3 scenarios
- **`final_comparison.png`** — Full metric panel

Architecture: **`/diagrams/asif_architecture.png`**

---

## Notebook

`notebooks/KPT_ASIF_Demo.ipynb` — complete step-by-step walkthrough:
1. Load and explore synthetic dataset
2. Apply signal engineering (smoothing, LRI, anomaly detection)
3. Demonstrate capacity adjustment
4. Train raw vs engineered models
5. Run full simulation
6. Visualise all metric improvements
7. Display architecture diagram inline

Open with: `jupyter notebook` or `jupyter lab`

---

## How the Code Maps to the System

```
generate_synthetic_data.py  →  Simulates realistic 218K orders with rush, promos, weather
signal_engine.smooth_signal()  →  Replaces raw FOR lag with EWMA de-noised signal
signal_engine.detect_anomalies()  →  Removes rush spike contamination from training labels
signal_engine.compute_live_rush_index()  →  Builds LRI: prep deviation + queue + spike signals
forecast_model.train_and_evaluate()  →  15-feature model; engineered features add ~20% MAE gain
signal_engine.capacity_adjusted_forecast()  →  Post-model scaling for high-load periods
simulation_runner.run_simulation()  →  End-to-end 3-scenario validation with CSV + plots
metrics.py  →  Operational metrics: MAE, RMSE, SLA%, rider idle, cancellation rate
```

---

*Python 3.x · numpy · pandas · scikit-learn · matplotlib · seaborn · xgboost (optional)*

*Modular, reproducible, and production-aligned.*
