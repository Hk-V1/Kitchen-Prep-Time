"""
generate_synthetic_data.py
--------------------------
Generates realistic time-series synthetic data for Kitchen Prep Time (KPT)
prediction across 10-50 restaurants. Saves output to /data/synthetic_orders.csv
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_restaurant_data(
    restaurant_id: int,
    n_days: int = 30,
    orders_per_day: int = 200,
) -> pd.DataFrame:
    """
    Generate synthetic order-level data for a single restaurant.

    Parameters
    ----------
    restaurant_id : int
        Unique identifier for the restaurant.
    n_days : int
        Number of days to simulate.
    orders_per_day : int
        Average number of orders per day.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        timestamp, restaurant_id, zomato_orders_per_min,
        promo_flag, weather_effect, local_event_flag,
        dine_in_load_proxy, sudden_rush_spike, true_prep_time,
        observed_prep_time
    """
    start_date = datetime(2024, 1, 1)
    records = []

    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        is_weekend = current_date.weekday() >= 5

        # Promo flag: 30% chance on weekends, 15% weekdays
        promo_flag = int(np.random.rand() < (0.30 if is_weekend else 0.15))

        # Local event: 10% chance
        local_event_flag = int(np.random.rand() < 0.10)

        # Weather effect: 0=clear, 1=light rain, 2=heavy rain
        weather_effect = np.random.choice([0, 1, 2], p=[0.60, 0.30, 0.10])

        # Simulate order timestamps throughout the day
        n_orders = int(np.random.normal(orders_per_day * (1.3 if is_weekend else 1.0), 20))
        n_orders = max(n_orders, 10)

        # Orders cluster around lunch (12-14h) and dinner (19-22h)
        lunch_orders = int(n_orders * 0.40)
        dinner_orders = int(n_orders * 0.45)
        other_orders = n_orders - lunch_orders - dinner_orders

        def sample_hour_minutes(mean_hour, std_h, count):
            hours = np.clip(np.random.normal(mean_hour, std_h, count), 0, 23)
            minutes = np.random.randint(0, 60, count)
            return hours, minutes

        lh, lm = sample_hour_minutes(13, 0.8, lunch_orders)
        dh, dm = sample_hour_minutes(20.5, 1.0, dinner_orders)
        oh = np.random.uniform(9, 23, other_orders)
        om = np.random.randint(0, 60, other_orders)

        all_hours = np.concatenate([lh, dh, oh])
        all_minutes = np.concatenate([lm, dm, om])

        for i in range(n_orders):
            hour = int(all_hours[i])
            minute = int(all_minutes[i])
            ts = current_date + timedelta(hours=hour, minutes=minute)

            # Zomato orders/min: peaks at meal times
            base_opm = 2 + 5 * np.exp(-0.5 * ((hour - 13) / 1.5) ** 2) + \
                       5 * np.exp(-0.5 * ((hour - 20.5) / 1.5) ** 2)
            zomato_opm = max(0, base_opm + np.random.normal(0, 0.5)
                             + (1.5 if promo_flag else 0)
                             + (1.0 if local_event_flag else 0))

            # Dine-in load proxy (0-1)
            dine_in_base = 0.3 + 0.4 * np.exp(-0.5 * ((hour - 13) / 2) ** 2) + \
                           0.4 * np.exp(-0.5 * ((hour - 20) / 2) ** 2)
            dine_in_load_proxy = np.clip(dine_in_base + np.random.normal(0, 0.05), 0, 1)

            # Sudden rush spike (binary, ~8% probability, higher at peak hours)
            rush_prob = 0.04 + 0.08 * (zomato_opm / 10)
            sudden_rush_spike = int(np.random.rand() < min(rush_prob, 0.25))

            # True prep time (minutes): base + contextual factors
            base_prep = np.random.normal(18, 3)
            true_prep_time = (
                base_prep
                + 4 * dine_in_load_proxy
                + 3 * (zomato_opm / 10)
                + 5 * sudden_rush_spike
                + 2 * (weather_effect == 2)
                + 1.5 * (weather_effect == 1)
                + 2 * promo_flag
                + 1.5 * local_event_flag
                + np.random.normal(0, 1.5)          # stochastic kitchen noise
            )
            true_prep_time = max(5, true_prep_time)

            # Observed prep time (noisy signal from FOR system)
            obs_noise = np.random.normal(0, 3) + (8 if sudden_rush_spike else 0)
            observed_prep_time = max(1, true_prep_time + obs_noise)

            records.append({
                "timestamp": ts,
                "restaurant_id": restaurant_id,
                "zomato_orders_per_min": round(zomato_opm, 3),
                "promo_flag": promo_flag,
                "weather_effect": weather_effect,
                "local_event_flag": local_event_flag,
                "dine_in_load_proxy": round(dine_in_load_proxy, 3),
                "sudden_rush_spike": sudden_rush_spike,
                "true_prep_time": round(true_prep_time, 2),
                "observed_prep_time": round(observed_prep_time, 2),
            })

    return pd.DataFrame(records)


def generate_all_restaurants(
    n_restaurants: int = 30,
    n_days: int = 30,
    output_path: str = "../data/synthetic_orders.csv",
) -> pd.DataFrame:
    """
    Generate data for multiple restaurants and save to CSV.

    Parameters
    ----------
    n_restaurants : int
        Number of restaurants to simulate (10-50).
    n_days : int
        Number of days per restaurant.
    output_path : str
        Path to save the CSV file.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame for all restaurants.
    """
    n_restaurants = np.clip(n_restaurants, 10, 50)
    print(f"Generating data for {n_restaurants} restaurants over {n_days} days...")

    all_dfs = []
    for rid in range(1, n_restaurants + 1):
        orders_per_day = np.random.randint(100, 300)
        df = generate_restaurant_data(rid, n_days=n_days, orders_per_day=orders_per_day)
        all_dfs.append(df)
        if rid % 10 == 0:
            print(f"  ✓ Restaurant {rid}/{n_restaurants} done")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\n✅ Dataset saved → {output_path}")
    print(f"   Shape : {combined.shape}")
    print(f"   Date range: {combined['timestamp'].min()} → {combined['timestamp'].max()}")
    return combined


if __name__ == "__main__":
    df = generate_all_restaurants(n_restaurants=30, n_days=30, output_path="../data/synthetic_orders.csv")
    print(df.head())
