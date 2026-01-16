# Project -1 : Linear Regression with Iris Dataset (MLflow Tracking) 

## 📌 Project Status
🚧 **Work in Progress**  
This project is under active development. Core setup and MLflow integration are completed, while model improvements and evaluation are still ongoing.

---

## 📖 Project Overview
This repository demonstrates a **Linear Regression model** built using the **Iris dataset**, with experiment tracking and model inference managed through **MLflow**.

The project focuses on:
- Setting up a clean Python development environment
- Training a regression model on the Iris dataset
- Tracking experiments using MLflow
- Logging and inferencing model artifacts with MLflow
- Inferencing model Artigacts with MLFLOW inferncing, Tracking parameters
- Comparing Diff models vs metrics
- Validate tyhe model before deployment via Inferencing,
- And load model back prediction as Generative python function (MLFLOW.pyfunc)
- Register model in MLFLOW: version, tags, and Aliase
- Inferencing from model registry: Model, parameters, ,model_uri path, prediction values

---

## 🛠️ Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- MLflow
- VS Code
- Virtual Environment (`venv`)

---

## Project -2 :   House Price Prediction with MLflow Tracking 

## 📌 Project Overview 
This project demonstrates an **end-to-end machine learning workflow** for **house price prediction** using the **California Housing dataset** and **MLflow** for experiment tracking, hyperparameter tuning, and model registration.

The main goals of this project are: 
- Train a regression model with hyperparameter tuning 
- Track all experiments, parameters, and metrics using MLflow
- Compare multiple runs in the MLflow UI 
- Register the best-performing model in the MLflow Model Registry 

--- 

## 📊 Dataset 
- **California Housing Dataset** 
- Source: `sklearn.datasets.fetch_california_housing` 
- Number of samples: **20,640**
- Features:  - MedInc (Median Income),  - HouseAge, - AveRooms, - AveBedrms 
  - Population, - AveOccup, - Latitude, - Longitude 
- Target: 
  - **Price** (Median house value in units of $100,000) 

---
## 🧠 Model 
- Algorithm: **Random Forest Regressor** 
- Evaluation metric: **Mean Squared Error (MSE)** 
- Hyperparameter tuning: **GridSearchCV** 

---

## ✅ Project Workflow 

### 1️⃣ Data Loading 
- Dataset loaded using `fetch_california_housing` 
- Converted into a pandas DataFrame 
- Target variable added as `Price` 

---

### 2️⃣ Data Preparation 
- Independent variables (`X`) created by dropping the `Price` column 
- Dependent variable (`y`) set as `Price` 
- Train-test split performed (80% training / 20% testing) 

---

### 3️⃣ Hyperparameter Tuning 
- Hyperparameter tuning implemented using `GridSearchCV` 
- Parameters tuned: 
  - `n_estimators` , - `max_depth` , - `min_samples_split`, - `min_samples_leaf`, - 3-fold cross-validation used 
- Scoring metric: `neg_mean_squared_error` 

---

### 4️⃣ Model Training & Evaluation 
- Best model selected from GridSearchCV 
- Predictions generated on test data 
- Mean Squared Error calculated 

---

### 5️⃣ MLflow Experiment Tracking 
- MLflow tracking server used (`http://127.0.0.1:5000`) 
- Logged to MLflow: Best hyperparameters, Mean Squared Error (MSE), Model artifacts  
- Model input/output signature inferred using `infer_signature` 

---

### 6️⃣ Model Registration 
- Best-performing model registered in **MLflow Model Registry** 
- Registered model name:


### Best Model: Random Forest Regressor 
    Best Hyperparameters: 
    n_estimators: 200 
    max_depth: None 
    min_samples_split: 2 
    min_samples_leaf: 1 
    Mean Squared Error (MSE): ~0.25 
    Model successfully tracked and registered in MLflow 
---
