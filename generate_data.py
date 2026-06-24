"""
generate_data.py
----------------
Generates a realistic synthetic sales dataset with 1500 records.
Run this once to create: dataset/sales_data.csv
"""

import pandas as pd
import numpy as np
import os

# Fix random seed so results are reproducible
np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_RECORDS = 1500

CATEGORIES  = ["Electronics", "Clothing", "Food", "Furniture", "Sports"]
REGIONS     = ["North", "South", "East", "West"]
SEASONS     = ["Spring", "Summer", "Autumn", "Winter"]

# Base revenue multipliers per category
CATEGORY_MULTIPLIER = {
    "Electronics": 1.8,
    "Clothing":    1.2,
    "Food":        0.9,
    "Furniture":   1.5,
    "Sports":      1.1,
}

# Region factor
REGION_MULTIPLIER = {
    "North": 1.0,
    "South": 1.1,
    "East":  0.95,
    "West":  1.2,
}

# Season factor
SEASON_MULTIPLIER = {
    "Spring": 1.0,
    "Summer": 1.15,
    "Autumn": 1.05,
    "Winter": 1.3,   # festive boost
}

# ── Generate columns ───────────────────────────────────────────────────────────
def generate_dataset():
    """Build and return the full synthetic DataFrame."""

    # Date range: Jan 2022 – Dec 2024
    dates = pd.date_range(start="2022-01-01", end="2024-12-31", periods=NUM_RECORDS)

    # Random categorical choices
    categories = np.random.choice(CATEGORIES, NUM_RECORDS)
    regions    = np.random.choice(REGIONS,    NUM_RECORDS)
    seasons    = np.random.choice(SEASONS,    NUM_RECORDS)

    # Continuous features
    marketing_spend    = np.random.randint(1000, 50000, NUM_RECORDS).astype(float)
    previous_month_sales = np.random.randint(5000, 100000, NUM_RECORDS).astype(float)
    units_sold         = np.random.randint(10, 500, NUM_RECORDS).astype(float)

    # ── Revenue formula (deterministic + noise) ────────────────────────────────
    revenue = []
    for i in range(NUM_RECORDS):
        base = (
            units_sold[i] * 150                        # price-per-unit proxy
            + marketing_spend[i] * 0.8                 # marketing ROI
            + previous_month_sales[i] * 0.3            # carry-over effect
        )
        # Apply multipliers
        base *= CATEGORY_MULTIPLIER[categories[i]]
        base *= REGION_MULTIPLIER[regions[i]]
        base *= SEASON_MULTIPLIER[seasons[i]]
        # Add realistic noise (±10 %)
        noise = np.random.uniform(0.90, 1.10)
        revenue.append(round(base * noise, 2))

    df = pd.DataFrame({
        "Date":                 dates.strftime("%Y-%m-%d"),
        "Product_Category":     categories,
        "Region":               regions,
        "Season":               seasons,
        "Marketing_Spend":      marketing_spend,
        "Previous_Month_Sales": previous_month_sales,
        "Units_Sold":           units_sold,
        "Revenue":              revenue,
    })

    return df


if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    df = generate_dataset()
    df.to_csv("dataset/sales_data.csv", index=False)
    print(f"✅  Dataset created → dataset/sales_data.csv  ({len(df)} rows)")
    print(df.head())
    print("\nBasic stats:")
    print(df.describe())
