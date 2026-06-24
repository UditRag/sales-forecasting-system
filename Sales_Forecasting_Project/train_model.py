"""
train_model.py
--------------
Trains a Random Forest Regressor inside a Scikit-learn Pipeline.

Steps
-----
1. Load dataset
2. Handle missing values
3. Separate features and target
4. Build a preprocessing + model Pipeline
5. Train and evaluate
6. Save pipeline as sales_pipeline.pkl

Run:  python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import OneHotEncoder, StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.ensemble         import RandomForestRegressor
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score


# ── 1. Load Data ──────────────────────────────────────────────────────────────
def load_data(path: str = "dataset/sales_data.csv") -> pd.DataFrame:
    """Load the CSV dataset into a Pandas DataFrame."""
    df = pd.read_csv(path)
    print(f"✅  Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ── 2. Prepare Features & Target ──────────────────────────────────────────────
def prepare_features(df: pd.DataFrame):
    """
    Drop unused columns, define X (features) and y (target).
    Returns X, y and the lists of categorical / numeric column names.
    """
    # Drop the Date column – it's not used directly in this simple model
    df = df.drop(columns=["Date"], errors="ignore")

    # Target variable
    TARGET = "Revenue"

    # Feature matrix
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Identify column types
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    print(f"   Categorical features : {cat_cols}")
    print(f"   Numerical  features  : {num_cols}")
    print(f"   Target               : {TARGET}")

    return X, y, cat_cols, num_cols


# ── 3. Build Pipeline ─────────────────────────────────────────────────────────
def build_pipeline(cat_cols: list, num_cols: list) -> Pipeline:
    """
    Create a full Scikit-learn Pipeline:
        Input → Impute → Encode / Scale → RandomForestRegressor
    """

    # ── Numerical sub-pipeline ────────────────────────────────────────────────
    # Step 1: Fill any missing numeric values with the column median
    # Step 2: Scale features so they are on a similar range
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # ── Categorical sub-pipeline ──────────────────────────────────────────────
    # Step 1: Fill missing text values with the most frequent category
    # Step 2: One-Hot encode so the model can use them
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # ── Combine both sub-pipelines ────────────────────────────────────────────
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,     num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    # ── Full Pipeline  ────────────────────────────────────────────────────────
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,    # 100 decision trees
            max_depth=None,      # trees grow until pure leaves
            random_state=42,     # reproducibility
            n_jobs=-1,           # use all CPU cores
        )),
    ])

    return pipeline


# ── 4. Train & Evaluate ───────────────────────────────────────────────────────
def train_and_evaluate(pipeline: Pipeline, X_train, X_test, y_train, y_test):
    """Fit the pipeline on training data and print evaluation metrics."""

    # Train
    pipeline.fit(X_train, y_train)
    print("\n✅  Model training complete.")

    # Predict on test set
    y_pred = pipeline.predict(X_test)

    # Metrics
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n── Evaluation Metrics ──────────────────────────────────")
    print(f"   MAE  (Mean Absolute Error)       : ₹ {mae:,.2f}")
    print(f"   RMSE (Root Mean Squared Error)   : ₹ {rmse:,.2f}")
    print(f"   R²   (Coefficient of Determination): {r2:.4f}  ({r2*100:.2f}%)")
    print("────────────────────────────────────────────────────────\n")

    return mae, rmse, r2, y_pred


# ── 5. Save Pipeline ──────────────────────────────────────────────────────────
def save_pipeline(pipeline: Pipeline, path: str = "sales_pipeline.pkl"):
    """Serialize the trained pipeline using Pickle."""
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"✅  Pipeline saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 56)
    print("   Sales Forecasting – Model Training")
    print("=" * 56)

    # 1. Load
    df = load_data()

    # 2. Prepare
    X, y, cat_cols, num_cols = prepare_features(df)

    # 3. Split  (80 % train | 20 % test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n   Train size : {len(X_train)}   Test size : {len(X_test)}")

    # 4. Build pipeline
    pipeline = build_pipeline(cat_cols, num_cols)

    # 5. Train + Evaluate
    mae, rmse, r2, y_pred = train_and_evaluate(
        pipeline, X_train, X_test, y_train, y_test
    )

    # 6. Save
    save_pipeline(pipeline)

    # Also save test results for the dashboard
    results_df = pd.DataFrame({
        "Actual":    y_test.values,
        "Predicted": y_pred,
    })
    results_df.to_csv("dataset/test_results.csv", index=False)
    print("✅  Test results saved → dataset/test_results.csv")

    return mae, rmse, r2


if __name__ == "__main__":
    main()
