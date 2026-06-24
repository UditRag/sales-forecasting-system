# 📈 Sales Forecasting System

> **Predicting Revenue with Data-Driven Precision**

---

## 📋 Project Overview

This project builds a complete **end-to-end Machine Learning system** to forecast
monthly sales revenue for a retail business. It uses a **Random Forest Regressor**
wrapped inside a **Scikit-learn Pipeline** and deployed as an interactive
**Streamlit web application**.

---

## 🗂️ Folder Structure

```
Sales_Forecasting_Project/
├── app.py                  ← Main Streamlit application (5 pages)
├── train_model.py          ← Pipeline training script
├── generate_data.py        ← Synthetic dataset generator
├── utils.py                ← Helper functions (charts, prediction, loaders)
├── sales_pipeline.pkl      ← Saved trained pipeline (auto-generated)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── dataset/
│   ├── sales_data.csv      ← Training dataset (auto-generated)
│   └── test_results.csv    ← Actual vs Predicted (auto-generated)
└── report/
    └── viva_qa.md          ← Viva questions & answers
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python 3.10+ |
| Machine Learning | Scikit-learn (Pipeline + RandomForestRegressor) |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |
| Model Storage | Pickle |

---

## 🚀 How to Run

### Step 1 – Clone / Download the project
```bash
cd Sales_Forecasting_Project
```

### Step 2 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 – Generate the dataset
```bash
python generate_data.py
```

### Step 4 – Train the model
```bash
python train_model.py
```

### Step 5 – Launch the Streamlit app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| MAE (Mean Absolute Error) | ₹ 13,312 |
| RMSE (Root Mean Squared Error) | ₹ 18,476 |
| R² Score | 0.9097 (≈ 91%) |

---

## 🔄 ML Pipeline Flow

```
Raw CSV Data
    │
    ▼
┌─────────────────────────────────────────────┐
│             ColumnTransformer               │
│  ┌──────────────────┐  ┌──────────────────┐ │
│  │  Numeric Branch  │  │ Categorical Branch│ │
│  │  SimpleImputer   │  │  SimpleImputer   │ │
│  │  StandardScaler  │  │  OneHotEncoder   │ │
│  └──────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────┘
    │
    ▼
RandomForestRegressor (100 trees)
    │
    ▼
Predicted Revenue (₹)
```

---

## 📱 Application Pages

| Page | Description |
|---|---|
| 🏠 Home | Project overview, objectives, workflow, quick stats |
| 🔮 Prediction | Interactive form → live revenue prediction |
| 📊 Dashboard | Charts: monthly trend, categories, distribution, actual vs predicted |
| 🧠 Model Info | Algorithm explanation, pipeline architecture, future scope |
| 🏋️ Train Model | Retrain the model from the UI with live metrics |

---

## 👨‍💻 Author
**Udit Raghuwanshi**  

