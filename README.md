# 🍽️ KPT Prediction – ASIF System
### Hackathon Submission: Kitchen Prep Time Prediction via Signal Redesign, De-noising, Live Rush Detection & Capacity-Aware Forecasting

---

## 📌 Problem Statement

Food delivery platforms suffer from inaccurate Kitchen Prep Time (KPT) forecasts due to:
- Noisy raw FOR (Food Order Received) signals polluted by rush spikes
- Lack of real-time kitchen load awareness
- Models ignoring external context (promos, weather, local events)
- No capacity-based adjustment to forecasts

Inaccurate KPT leads to: early rider dispatch → idle wait time → bad customer experience → order cancellations.

---

## 🚀 ASIF System Overview

**ASIF = Adaptive Signal-Integrated Forecast**

```
Raw Signals → Signal De-noising → Live Rush Index → ML Forecast → Capacity Adjustment → Dispatch
```

| Stage | What we do |
|-------|-----------|
| Signal De-noising | EWMA smoothing + anomaly detection strips rush noise |
| Live Rush Index (LRI) | Composite score combining prep deviation, queue depth, spike rate |
| Multi-signal ML | XGBoost trained on engineered features (internal + external) |
| Capacity Adjustment | Scales forecast up/down based on dine-in + queue load |

---

## 📁 Folder Structure

```
kpt_hackathon/
├── data/
│   ├── synthetic_orders.csv        # Generated dataset (30 restaurants × 30 days)
│   ├── simulation_results.csv      # Scenario comparison output
│   └── plots/                      # Generated charts
├── src/
│   ├── generate_synthetic_data.py  # Dataset generation
│   ├── signal_engine.py            # Signal processing functions
│   ├── forecast_model.py           # XGBoost forecast model
│   ├── simulation_runner.py        # 3-scenario simulation
│   ├── metrics.py                  # MAE, RMSE, SLA%, idle, cancellation
│   └── generate_architecture_diagram.py
├── notebooks/
│   └── KPT_ASIF_Demo.ipynb         # Full step-by-step demo
├── diagrams/
│   └── asif_architecture.png       # System architecture diagram
└── README.md
```

---

## ⚙️ Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost
```

> XGBoost is optional – the system falls back to sklearn's GradientBoostingRegressor automatically.

---

## 🏃 How to Run

### Step 1: Generate Synthetic Data
```bash
cd src
python generate_synthetic_data.py
# → Saves: ../data/synthetic_orders.csv
```

### Step 2: Run Signal Engine (standalone test)
```bash
python signal_engine.py
# → Prints signal-engineered sample output
```

### Step 3: Train Forecast Models
```bash
python forecast_model.py
# → Trains Raw-only and ASIF models, prints MAE/RMSE comparison
```

### Step 4: Run Full Simulation
```bash
python simulation_runner.py
# → Runs all 3 scenarios, prints table, saves plots
```

### Step 5: View Notebook Demo
```bash
cd ../notebooks
jupyter notebook KPT_ASIF_Demo.ipynb
```

### Step 6: Generate Architecture Diagram
```bash
cd ../src
python generate_architecture_diagram.py
# → Saves: ../diagrams/asif_architecture.png
```

---

## 📊 Key Results (Typical Run)

| Scenario | MAE (min) | RMSE (min) | SLA % | Rider Idle | Cancel Rate |
|----------|-----------|------------|-------|------------|-------------|
| Baseline (noisy FOR) | ~7.0 | ~9.5 | ~58% | ~4.5 min | ~22% |
| ML-Only (raw features) | ~4.5 | ~6.1 | ~71% | ~2.8 min | ~14% |
| **ASIF System** | **~2.2** | **~3.1** | **~87%** | **~1.1 min** | **~6%** |

**Improvements over baseline:**
- ✅ MAE reduced by **~68%**
- ✅ SLA compliance up by **~29 percentage points**
- ✅ Rider idle time cut by **~76%**
- ✅ Cancellation rate reduced by **~73%**

---

## 🔬 Core Modules

### `signal_engine.py`
| Function | Description |
|----------|-------------|
| `smooth_signal(signal, alpha)` | EWMA de-noising |
| `rolling_variance(signal, window)` | Kitchen instability quantification |
| `detect_anomalies(signal, threshold)` | Rolling Z-score anomaly flags |
| `compute_live_rush_index(data)` | Composite LRI (0–1 score) |
| `capacity_adjusted_forecast(forecast, capacity_pct)` | Load-aware scaling |

### `metrics.py`
| Function | What it measures |
|----------|----------------|
| `mae(y_true, y_pred)` | Mean Absolute Error |
| `rmse(y_true, y_pred)` | Root Mean Squared Error |
| `sla_percentage(...)` | % orders within ±5 min |
| `rider_idle_proxy(...)` | Avg minutes rider waits |
| `cancellation_proxy(...)` | % over-predicted by >8 min |

---

## 📈 Visualisations

After running the simulation, plots are saved in `/data/plots/`:
- `signal_denoising.png` – Raw vs smoothed + LRI
- `signal_analysis.png` – Variance + anomaly overlay
- `feature_importance.png` – ASIF model feature ranking
- `pred_vs_actual.png` – Scatter plot comparison
- `simulation_comparison.png` – Bar chart of all scenarios
- `final_comparison.png` – Full metrics comparison

Architecture diagram: `/diagrams/asif_architecture.png`

---

## 💡 Design Decisions

1. **EWMA over simple MA** – Lower lag at rush onset, critical for real-time dispatch
2. **LRI as a feature, not just a filter** – Feeds directly into ML model, adds contextual awareness
3. **Capacity adjustment post-ML** – Decouples prediction from operational state, more interpretable
4. **Anomaly detection before training** – Prevents rush spike contamination of training labels
5. **Multi-source signals** – Weather + promo + local events as first-class features, not afterthoughts

---

## 📓 Notebook

`notebooks/KPT_ASIF_Demo.ipynb` walks through every step with inline plots and commentary. Open with Jupyter Notebook or JupyterLab.

