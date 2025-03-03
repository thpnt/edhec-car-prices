# Car Price Prediction Project

This repository contains a car price prediction pipeline and a FastAPI service for serving model predictions. The workflow includes:

1. **Preprocessing & Feature Engineering**  
2. **Exploratory Data Analysis (EDA)**  
3. **Correlation and Iterative VIF Reduction**  
4. **Model Comparison and Hyperparameter Tuning**  
5. **Model Training & Packaging**  
6. **FastAPI Deployment**

---

## 1. Preprocessing Pipeline

**Overview**  
- Cleans and formats raw data (e.g., converting string mileage to float).  
- Uses encoders (OneHotEncoder / OrdinalEncoder) for categorical variables.  
- Imputes missing numeric values with KNNImputer.  
- Creates interaction features (e.g., Fuel type × Cylinders).  
- Applies Iterative VIF reduction to remove highly collinear features.

**Files & Modules**  
- **`preproc/cleaning.py`** for data cleaning (string-to-float conversions, turbo flag, etc.).  
- **`preproc/preprocessing.py`** for custom transformers and pipeline creation.  

**Usage**  
- Load, clean, split data into features & target.  
- Build and fit the pipeline, then transform your training data into a preprocessed DataFrame suitable for modeling.

---

## 2. Correlation & Iterative VIF Analysis

During EDA, correlation analysis is performed:
- **Pearson Correlation** for continuous features.  
- **Alternative measures** (e.g., Eta / rank-based) for categorical vs. continuous.  
- **Iterative VIF**: Features are dropped one at a time if their VIF exceeds a threshold (e.g., 10).

This process ensures reduced redundancy and more stable modeling.

---

## 3. Model Comparison & Hyperparameter Tuning

Models typically considered include:
- **Linear Models** (LinearRegression, Ridge, Lasso).  
- **Tree Ensembles** (RandomForest, GradientBoosting, XGBoost, AdaBoost).  
- **SVR** (Support Vector Regression).

**Workflow**  
1. Compare baseline metrics (RMSE, MAE) via cross-validation.  
2. Fine-tune the top performers with GridSearchCV or RandomizedSearchCV.  
3. Conduct residual analysis and feature-importance inspection.

---

## 4. Model Training Script

**Steps**  
1. Load the cleaned and preprocessed data.  
2. Define and train the final model (e.g., RandomForestRegressor).  
3. Save the fitted pipeline and model using joblib.

These files can be used later for inference in the FastAPI application.

---

## 5. FastAPI Service

**Project Structure**  
Typically organized as follows:
- `api/` with `app.py` for the FastAPI endpoints.  
- `preproc/` for cleaning functions, custom transformers, pipeline creation.  
- `models/` for the serialized pipeline and model files.  
- `data/` for training data and any supplemental CSVs.  
- `notebooks/` for exploration and comparison.

**Endpoints**  
- **`/predict`**: Accepts a single JSON sample of features, returns one prediction.  
- **`/batch_predict`**: Accepts multiple samples, returns a list of predictions.

The API automatically generates interactive documentation at `/docs` (Swagger UI) and `/redoc`.

---

## 6. Deployment

**Local**  
- Install dependencies from `requirements.txt`.  
- Run with uvicorn, then visit `127.0.0.1:8000/docs`.
