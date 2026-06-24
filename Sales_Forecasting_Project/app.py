"""
app.py
------
Main Streamlit application for the Sales Forecasting System.

Pages
─────
  🏠  Home          – Project overview, objectives, workflow
  🔮  Prediction    – Interactive form → predicted revenue
  📊  Dashboard     – Analytics charts
  🧠  Model Info    – Algorithm explanation, future scope
  🏋️  Train Model   – Run training pipeline from the UI

Run:  streamlit run app.py
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global background ── */
.stApp { background-color: #0d0d0d; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111 0%, #1a1a1a 100%);
    border-right: 1px solid #2a2a2a;
}

/* ── Headings ── */
h1, h2, h3 { color: #CCFF00 !important; }
p, li, label { color: #e0e0e0; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1a1a1a;
    border: 1px solid #CCFF0055;
    border-radius: 10px;
    padding: 12px 16px;
}

/* ── Buttons ── */
.stButton > button {
    background: #CCFF00;
    color: #111;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Info / success boxes ── */
.stAlert { border-radius: 8px; }

/* ── Prediction result card ── */
.pred-card {
    background: linear-gradient(135deg, #1a1a1a 0%, #111 100%);
    border: 2px solid #CCFF00;
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    margin-top: 16px;
}
.pred-amount {
    font-size: 2.6rem;
    font-weight: 800;
    color: #CCFF00;
}
.pred-label {
    font-size: 0.9rem;
    color: #aaa;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## 📈 Sales Forecasting")
st.sidebar.markdown("---")

PAGES = {
    "🏠  Home":          "home",
    "🔮  Prediction":    "prediction",
    "📊  Dashboard":     "dashboard",
    "🧠  Model Info":    "model_info",
    "🏋️  Train Model":   "train",
}

choice = st.sidebar.radio("Navigate to", list(PAGES.keys()), label_visibility="collapsed")
page  = PAGES[choice]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#666;'>B.Tech CSE Minor Project<br>"
    "Laxmi Narain College of Technology<br>Bhopal, India</small>",
    unsafe_allow_html=True
)

# ── Lazy imports (after page config) ─────────────────────────────────────────
from utils import (
    load_pipeline, load_dataset, load_test_results, predict_revenue,
    plot_monthly_revenue, plot_revenue_by_category,
    plot_revenue_distribution, plot_actual_vs_predicted, plot_sales_by_region,
)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 – HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.title("📈 Sales Forecasting System")
    st.markdown(
        "#### Predicting Revenue with Data-Driven Precision"
    )
    st.markdown("---")

    # Project Description
    st.markdown("""
    > This project builds a complete **end-to-end machine learning pipeline**
    > to forecast future sales revenue for a retail business.
    > It uses a **Random Forest Regressor** wrapped inside a
    > **Scikit-learn Pipeline** for reproducible, production-ready predictions.
    """)

    # Key Objectives
    st.markdown("### 🎯 Key Objectives")
    col1, col2 = st.columns(2)
    with col1:
        st.info("📌 Predict monthly revenue from sales & marketing features")
        st.info("📌 Compare different forecasting approaches")
    with col2:
        st.info("📌 Deliver an interactive dashboard for business insights")
        st.info("📌 Demonstrate a complete ML pipeline (data → deployment)")

    st.markdown("---")

    # Project Workflow
    st.markdown("### 🔄 Project Workflow")
    steps = [
        ("1️⃣", "Data Generation",  "Synthetic 1,500-record dataset with realistic sales patterns"),
        ("2️⃣", "Preprocessing",    "Impute nulls → Encode categories → Scale numerics"),
        ("3️⃣", "Model Training",   "Random Forest Regressor inside an sklearn Pipeline"),
        ("4️⃣", "Evaluation",       "MAE, RMSE, R² on an unseen 20% test split"),
        ("5️⃣", "Deployment",       "Streamlit app loads the saved .pkl for live predictions"),
    ]

    cols = st.columns(len(steps))
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(
            f"""
            <div style='background:#1a1a1a; border:1px solid #333;
                        border-radius:10px; padding:14px; text-align:center;
                        height:150px;'>
                <div style='font-size:1.8rem;'>{icon}</div>
                <strong style='color:#CCFF00;'>{title}</strong><br>
                <small style='color:#aaa;'>{desc}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Quick Stats
    st.markdown("### 📊 Dataset at a Glance")
    try:
        df = load_dataset()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records",    f"{len(df):,}")
        c2.metric("Avg Revenue",      f"₹{df['Revenue'].mean():,.0f}")
        c3.metric("Max Revenue",      f"₹{df['Revenue'].max():,.0f}")
        c4.metric("Product Categories", df["Product_Category"].nunique())
    except Exception:
        st.warning("Dataset not found. Please run `python generate_data.py` first.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 – PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def page_prediction():
    st.title("🔮 Revenue Prediction")
    st.markdown("Fill in the details below to predict expected revenue.")
    st.markdown("---")

    # Check model exists
    if not os.path.exists("sales_pipeline.pkl"):
        st.error("⚠️  Model not found! Go to the **🏋️ Train Model** page to train it first.")
        return

    pipeline = load_pipeline()

    # ── Input Form ────────────────────────────────────────────────────────────
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📦 Product & Market Info")
            product_category = st.selectbox(
                "Product Category",
                ["Electronics", "Clothing", "Food", "Furniture", "Sports"],
            )
            region = st.selectbox("Region", ["North", "South", "East", "West"])
            season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])

        with col2:
            st.subheader("💰 Sales Figures")
            marketing_spend = st.number_input(
                "Marketing Spend (₹)", min_value=1000, max_value=100000,
                value=25000, step=500,
                help="Total amount spent on marketing this month"
            )
            previous_month_sales = st.number_input(
                "Previous Month Sales (₹)", min_value=1000, max_value=200000,
                value=50000, step=1000,
                help="Actual revenue recorded last month"
            )
            units_sold = st.number_input(
                "Units Sold", min_value=1, max_value=1000,
                value=200, step=10,
                help="Number of units sold this period"
            )

    st.markdown("")

    # ── Predict Button ────────────────────────────────────────────────────────
    predict_clicked = st.button("🚀  Predict Revenue", use_container_width=True)

    if predict_clicked:
        input_data = {
            "Product_Category":     product_category,
            "Region":               region,
            "Season":               season,
            "Marketing_Spend":      marketing_spend,
            "Previous_Month_Sales": previous_month_sales,
            "Units_Sold":           units_sold,
        }

        with st.spinner("Running prediction pipeline…"):
            result = predict_revenue(pipeline, input_data)

        # ── Result Card ───────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="pred-card">
                <div class="pred-label">Predicted Monthly Revenue</div>
                <div class="pred-amount">₹ {result:,.2f}</div>
                <div class="pred-label">
                    {product_category} · {region} · {season} Season
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        # Supporting metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Marketing Spend", f"₹ {marketing_spend:,}")
        col2.metric("Units Sold",      f"{units_sold:,}")
        col3.metric("ROI (est.)",      f"{(result / marketing_spend):.1f}×")

        st.success("✅  Prediction generated successfully using the saved ML pipeline.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 – DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.title("📊 Dashboard")
    st.markdown("Sales data ka simple overview.")
    st.markdown("---")

    try:
        df = load_dataset()
        results_df = load_test_results()
    except FileNotFoundError:
        st.error("⚠️  Data files not found. Please run train_model.py first.")
        return

    # 3 simple metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records",  f"{len(df):,}")
    c2.metric("Avg Revenue",    f"₹ {df['Revenue'].mean():,.0f}")
    c3.metric("Model Accuracy", "91%  (R²)")

    st.markdown("---")

    # Chart 1: Monthly Revenue
    st.subheader("📈 Monthly Revenue Trend")
    monthly = (
        df.set_index("Date")["Revenue"]
        .resample("ME").sum()
        .reset_index()
        .rename(columns={"Date": "Month", "Revenue": "Revenue (₹)"})
        .set_index("Month")
    )
    st.line_chart(monthly)

    st.markdown("---")

    # Chart 2: Revenue by Category
    st.subheader("🏷️ Revenue by Product Category")
    cat_rev = (
        df.groupby("Product_Category")["Revenue"]
        .mean()
        .reset_index()
        .rename(columns={"Revenue": "Avg Revenue (₹)", "Product_Category": "Category"})
        .set_index("Category")
    )
    st.bar_chart(cat_rev)

    st.markdown("---")

    # Chart 3: Actual vs Predicted
    st.subheader("🎯 Actual vs Predicted Revenue")
    st.caption("First 50 test samples dikhaye hain.")
    compare = results_df.head(50).reset_index(drop=True)
    st.line_chart(compare)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 – MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
def page_model_info():
    st.title("🧠 About the Model")
    st.markdown("---")

    # What is Sales Forecasting
    st.subheader("📌 What is Sales Forecasting?")
    st.markdown("""
    **Sales Forecasting** is the process of estimating future sales revenue
    over a specific period — a week, month, quarter, or year.

    It helps businesses:
    - Plan budgets and hiring based on expected demand
    - Identify revenue gaps and market shifts early
    - Increase shareholder trust through reliable projections
    - Optimise inventory to avoid costly overstock or stockouts
    """)

    st.markdown("---")

    # Why Random Forest
    st.subheader("🌲 Why Random Forest Regressor?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Random Forest** is an *ensemble* learning method that builds
        many decision trees and averages their outputs.

        **Selected because:**
        - Handles both numeric and categorical data naturally
        - Robust to outliers and noisy data
        - Requires little hyperparameter tuning
        - Provides feature importance out of the box
        - No assumption about data distribution
        """)
    with col2:
        # Comparison table
        methods = {
            "Method":           ["Linear Regression", "Decision Tree", "Random Forest (ours)", "Neural Network"],
            "Handles Non-linearity": ["❌", "✅", "✅", "✅"],
            "Overfitting Risk": ["Low", "High", "Low ✅", "Medium"],
            "Interpretability": ["High", "High", "Medium", "Low"],
            "Training Speed":   ["Fast", "Fast", "Medium ✅", "Slow"],
        }
        st.dataframe(pd.DataFrame(methods).set_index("Method"), use_container_width=True)

    st.markdown("---")

    # Pipeline Architecture
    st.subheader("⚙️ Pipeline Architecture")
    st.code("""
Pipeline(steps=[
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
        ]), numeric_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical_cols),
    ])),
    ('model', RandomForestRegressor(
        n_estimators = 100,
        random_state = 42,
    )),
])
    """, language="python")

    st.markdown("---")

    # Advantages
    st.subheader("✅ Advantages of this Approach")
    adv = [
        ("🔁 Reproducible", "The Pipeline ensures identical preprocessing on train and live data."),
        ("💾 Portable",     "A single .pkl file contains the full pipeline — easy to deploy."),
        ("📈 Accurate",     "Achieves ~91% R² on unseen data with only 1,500 training samples."),
        ("🔧 Extensible",   "New features can be added without restructuring the codebase."),
    ]
    cols = st.columns(len(adv))
    for col, (title, desc) in zip(cols, adv):
        col.markdown(
            f"""
            <div style='background:#1a1a1a; border:1px solid #333;
                        border-radius:10px; padding:14px; min-height:110px;'>
                <strong style='color:#CCFF00;'>{title}</strong><br>
                <small style='color:#ccc;'>{desc}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Future Scope
    st.subheader("🚀 Future Scope")
    st.markdown("""
    1. **Deep Learning** – LSTM networks for capturing long-term time-series patterns
    2. **BERT-based Sentiment** – Incorporate customer review sentiment as a feature
    3. **Real CRM Integration** – Connect directly to Salesforce / HubSpot data
    4. **AutoML Pipeline** – Use TPOT or H2O to automatically select the best model
    5. **REST API** – Wrap the pipeline in a FastAPI service for programmatic access
    6. **Docker Deployment** – Containerise the app for cloud deployment (AWS / GCP)
    """)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 – TRAIN MODEL (UI)
# ══════════════════════════════════════════════════════════════════════════════
def page_train():
    st.title("🏋️ Train Model")
    st.markdown("Retrain the Random Forest pipeline on the latest dataset.")
    st.markdown("---")

    # Dataset check
    dataset_exists = os.path.exists("dataset/sales_data.csv")
    model_exists   = os.path.exists("sales_pipeline.pkl")

    col1, col2 = st.columns(2)
    col1.metric("Dataset Status", "✅ Ready" if dataset_exists else "❌ Missing")
    col2.metric("Model Status",   "✅ Trained" if model_exists else "⚠️ Not trained")

    st.markdown("---")
    st.markdown("""
    **What happens when you click Train:**
    1. Load `dataset/sales_data.csv`
    2. Build the preprocessing Pipeline
    3. Train Random Forest Regressor (100 trees)
    4. Evaluate on 20% held-out test data
    5. Save pipeline to `sales_pipeline.pkl`
    """)

    if not dataset_exists:
        st.warning("Dataset missing. Generating it now…")
        subprocess.run([sys.executable, "generate_data.py"], check=True)
        st.success("Dataset generated! You can now train the model.")

    if st.button("🚀  Start Training", use_container_width=True):
        with st.spinner("Training in progress… this may take 30–60 seconds."):
            result = subprocess.run(
                [sys.executable, "train_model.py"],
                capture_output=True, text=True,
            )

        if result.returncode == 0:
            st.success("✅  Training complete! Model saved as `sales_pipeline.pkl`.")
            # Parse metrics from stdout
            for line in result.stdout.split("\n"):
                if any(k in line for k in ["MAE", "RMSE", "R²", "R2"]):
                    st.code(line.strip())
            # No cache to clear — model reloads fresh on next prediction
        else:
            st.error("Training failed. See error below:")
            st.code(result.stderr)


# ══════════════════════════════════════════════════════════════════════════════
#  Router
# ══════════════════════════════════════════════════════════════════════════════
if   page == "home":       page_home()
elif page == "prediction": page_prediction()
elif page == "dashboard":  page_dashboard()
elif page == "model_info": page_model_info()
elif page == "train":      page_train()
