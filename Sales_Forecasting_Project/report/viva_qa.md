# 🎓 Viva Questions & Answers
## Sales Forecasting System — B.Tech CSE Minor Project

---

### Section 1: Project Basics

**Q1. What is the objective of your project?**

> The objective is to build a machine learning system that predicts monthly sales revenue for a retail business based on features like product category, region, season, marketing spend, previous month's sales, and units sold. The system is deployed as a Streamlit web application.

---

**Q2. What dataset did you use?**

> We generated a synthetic dataset of 1,500 records using NumPy and Pandas. Each record represents one sales transaction with 7 features and one target variable (Revenue). The data includes realistic patterns like seasonal boosts and category-wise multipliers.

---

**Q3. What is the target variable in your project?**

> The target variable is **Revenue** — the monthly sales revenue in Indian Rupees (₹).

---

### Section 2: Machine Learning Concepts

**Q4. What is a Random Forest Regressor?**

> Random Forest is an ensemble learning algorithm that builds multiple decision trees during training and outputs the average of their individual predictions. The "forest" of trees reduces variance and overfitting compared to a single decision tree.

---

**Q5. Why did you choose Random Forest over Linear Regression?**

> Linear Regression assumes a linear relationship between features and target. Our data has non-linear interactions (e.g., a seasonal spike combined with high marketing spend in Electronics gives disproportionately higher revenue). Random Forest captures these non-linear patterns naturally.

---

**Q6. What is an Ensemble Method?**

> An ensemble method combines multiple machine learning models to produce a better predictive result than any single model. Random Forest is a **bagging** ensemble — it trains trees on random subsets of data and averages predictions.

---

**Q7. What does `n_estimators=100` mean?**

> It means the Random Forest will build 100 individual decision trees. Each tree is trained on a random bootstrap sample of the training data. More trees generally mean better accuracy but slower training.

---

**Q8. What is `random_state=42`?**

> `random_state` sets a seed for the random number generator, ensuring that the same results are produced every time the code runs. This makes experiments **reproducible**.

---

### Section 3: Scikit-learn Pipeline

**Q9. What is a Scikit-learn Pipeline?**

> A Pipeline chains multiple processing steps (transformers and a final estimator) into a single object. It ensures that the same transformations are applied in the same order during both training and prediction, preventing data leakage.

---

**Q10. What is data leakage and how does the Pipeline prevent it?**

> Data leakage happens when information from the test set influences the training process (e.g., computing the mean for imputation using test data). A Pipeline fits all transformers only on training data and applies learned parameters to test data, preventing leakage.

---

**Q11. What is a ColumnTransformer?**

> `ColumnTransformer` applies different preprocessing pipelines to different subsets of columns. In our project, it applies `StandardScaler` to numeric columns and `OneHotEncoder` to categorical columns simultaneously.

---

**Q12. What is OneHotEncoding?**

> One-Hot Encoding converts categorical text values into binary columns. For example, `Region = North` becomes `[1, 0, 0, 0]` for (North, South, East, West). Machine learning models can only work with numbers, so this conversion is necessary.

---

**Q13. What is StandardScaler? Why is it needed?**

> `StandardScaler` transforms numeric features so they have mean = 0 and standard deviation = 1. This is important because features like `Marketing_Spend` (range: 1K–50K) and `Units_Sold` (range: 10–500) are on very different scales. While Random Forests are not affected by scale, we include it for good pipeline practice.

---

**Q14. What is SimpleImputer?**

> `SimpleImputer` fills in missing values (NaN) in the dataset. We use `strategy="median"` for numeric columns (robust to outliers) and `strategy="most_frequent"` for categorical columns.

---

### Section 4: Evaluation Metrics

**Q15. What metrics did you use to evaluate the model?**

> We used three regression metrics:
> - **MAE (Mean Absolute Error)** – Average absolute difference between actual and predicted values. Easy to interpret in rupees.
> - **RMSE (Root Mean Squared Error)** – Square root of average squared errors. Penalises large errors more heavily.
> - **R² Score** – Proportion of variance in Revenue explained by the model. Our R² is ~0.91, meaning the model explains 91% of the variance.

---

**Q16. What does an R² of 0.91 mean?**

> It means our model explains **91% of the variation** in the actual revenue values. The remaining 9% is unexplained — caused by factors not in our dataset (e.g., competitor actions, economic news). An R² above 0.85 is generally considered good for a regression model.

---

**Q17. Why is RMSE higher than MAE?**

> RMSE squares the errors before averaging, which gives higher weight to large prediction errors. It is always ≥ MAE. When RMSE is noticeably larger than MAE, it indicates the model is making some large errors on a few records.

---

### Section 5: Streamlit & Deployment

**Q18. What is Streamlit?**

> Streamlit is an open-source Python library that converts Python scripts into interactive web applications without requiring any front-end (HTML/CSS/JavaScript) knowledge. It is widely used for ML dashboards and demos.

---

**Q19. How does the Prediction page work?**

> The user fills in an input form (category, region, season, marketing spend, previous sales, units sold). When the Predict button is clicked, the inputs are assembled into a Pandas DataFrame, passed to the loaded Pipeline, and the model's output (predicted revenue) is displayed as a styled metric card.

---

**Q20. What is Pickle and why is it used?**

> Pickle is Python's built-in module for serialising (saving) and deserialising (loading) Python objects to/from files. We use it to save the trained sklearn Pipeline as `sales_pipeline.pkl` so it can be loaded later without retraining.

---

**Q21. What is `@st.cache_resource`?**

> It is a Streamlit decorator that caches the return value of a function so it is computed only once per session. We use it on `load_pipeline()` so the .pkl file is read from disk only once, making the app faster.

---

### Section 6: Data & Features

**Q22. What features does your model use?**

> - **Product_Category** (categorical) – Electronics, Clothing, Food, Furniture, Sports
> - **Region** (categorical) – North, South, East, West
> - **Season** (categorical) – Spring, Summer, Autumn, Winter
> - **Marketing_Spend** (numeric) – Budget spent on marketing (₹)
> - **Previous_Month_Sales** (numeric) – Last month's actual revenue (₹)
> - **Units_Sold** (numeric) – Number of units sold in the period

---

**Q23. Why was the Date column dropped?**

> In our simplified model we do not use the Date directly as a feature (time-series modelling would require lag features, rolling averages etc.). Instead, we use `Season` as a proxy for temporal patterns, which captures most of the seasonal variation.

---

**Q24. What is the train-test split ratio you used?**

> We used an 80/20 split — 1,200 records for training and 300 records for testing. The `random_state=42` ensures the split is reproducible.

---

### Section 7: Future Scope

**Q25. How can this project be improved?**

> 1. Use real sales data from a CRM (Salesforce, Zoho)
> 2. Add LSTM for true time-series modelling
> 3. Incorporate external features: economic index, competitor pricing, weather
> 4. Deploy as a REST API using FastAPI
> 5. Containerise with Docker for cloud deployment
> 6. Use SHAP values to explain individual predictions (model interpretability)
> 7. Add user authentication and multi-tenant support
