"""
utils.py
--------
Shared helper functions used by app.py.
Keeps the main app file clean and readable.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st


# ── Model Loading ─────────────────────────────────────────────────────────────
def load_pipeline(path: str = "sales_pipeline.pkl"):
    """Load and return the trained sklearn Pipeline from disk."""
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


# ── Dataset Loading ───────────────────────────────────────────────────────────
def load_dataset(path: str = "dataset/sales_data.csv") -> pd.DataFrame:
    """Read the sales CSV and parse the Date column. No cache — always fresh."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def load_test_results(path: str = "dataset/test_results.csv") -> pd.DataFrame:
    """Read Actual vs Predicted results. No cache — always fresh."""
    return pd.read_csv(path)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_revenue(pipeline, input_dict: dict) -> float:
    """
    Build a single-row DataFrame from user inputs and
    return the predicted revenue (₹).
    """
    input_df = pd.DataFrame([input_dict])
    prediction = pipeline.predict(input_df)[0]
    return round(prediction, 2)


# ── Chart Helpers ─────────────────────────────────────────────────────────────

def set_dark_style():
    """Apply a consistent dark background style to all Matplotlib figures."""
    plt.style.use("dark_background")


def plot_monthly_revenue(df: pd.DataFrame):
    """Line chart: Monthly Revenue Trend."""
    set_dark_style()

    monthly = (
        df.set_index("Date")["Revenue"]
        .resample("ME")           # month-end frequency
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["Date"], monthly["Revenue"], color="#CCFF00",
            linewidth=2, marker="o", markersize=4)
    ax.fill_between(monthly["Date"], monthly["Revenue"],
                    alpha=0.15, color="#CCFF00")
    ax.set_title("Monthly Revenue Trend", fontsize=14, pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1_000:.0f}K"
    ))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_revenue_by_category(df: pd.DataFrame):
    """Bar chart: Revenue by Product Category."""
    set_dark_style()

    cat_rev = (
        df.groupby("Product_Category")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(cat_rev.index, cat_rev.values, color="#CCFF00", alpha=0.85,
                  edgecolor="#111", linewidth=0.7)
    ax.set_title("Total Revenue by Product Category", fontsize=14, pad=12)
    ax.set_xlabel("Category")
    ax.set_ylabel("Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1_000_000:.1f}M"
    ))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    # Add value labels on bars
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"₹{bar.get_height()/1_000_000:.2f}M",
            ha="center", va="bottom", fontsize=8, color="white"
        )
    fig.tight_layout()
    return fig


def plot_revenue_distribution(df: pd.DataFrame):
    """Histogram: Distribution of Revenue values."""
    set_dark_style()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["Revenue"], bins=40, color="#CCFF00", alpha=0.75,
            edgecolor="#111", linewidth=0.5)
    ax.set_title("Revenue Distribution", fontsize=14, pad=12)
    ax.set_xlabel("Revenue (₹)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1_000:.0f}K"
    ))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(results_df: pd.DataFrame):
    """Scatter plot: Actual Revenue vs Predicted Revenue."""
    set_dark_style()

    actual    = results_df["Actual"]
    predicted = results_df["Predicted"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(actual, predicted, color="#CCFF00", alpha=0.5,
               s=18, edgecolors="none")

    # Perfect-prediction reference line
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], color="white", linewidth=1.2,
            linestyle="--", label="Perfect Prediction")

    ax.set_title("Actual vs Predicted Revenue", fontsize=14, pad=12)
    ax.set_xlabel("Actual Revenue (₹)")
    ax.set_ylabel("Predicted Revenue (₹)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1_000:.0f}K"
    ))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1_000:.0f}K"
    ))
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_sales_by_region(df: pd.DataFrame):
    """Horizontal bar chart: Units Sold by Region."""
    set_dark_style()

    region_sales = (
        df.groupby("Region")["Units_Sold"]
        .sum()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(region_sales.index, region_sales.values,
            color="#CCFF00", alpha=0.85, edgecolor="#111", linewidth=0.7)
    ax.set_title("Total Units Sold by Region", fontsize=14, pad=12)
    ax.set_xlabel("Units Sold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig
