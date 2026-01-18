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

# Project -3 ANN with MLflow – End‑to‑End MLOps Project

## 🔍 Project Summary
Production-oriented **MLOps pipeline** demonstrating how to train, tune, track, register, and serve an **Artificial Neural Network (ANN)** using **MLflow**.  
The project covers the **full ML lifecycle** from experimentation to deployment-ready inference.

---

## 💼 Why This Project Matters  
This project demonstrates hands-on experience with:
- ✅ **Experiment tracking at scale**
- ✅ **Hyperparameter optimization**
- ✅ **Model registry & versioning**
- ✅ **Reproducible ML workflows**
- ✅ **Deployment-ready model artifacts**

It reflects **real-world ML engineering and MLOps practices**, not just model training.

---

## 🧠 Technical Highlights

### Model
- ANN built with **Keras**
- Regression task (Wine Quality prediction)
- Feature normalization inside the model graph
- Metric-driven model selection (RMSE)

### Optimization
- **Hyperopt + TPE** for hyperparameter search
- Search space:
  - Learning rate (log-uniform)
  - Momentum (uniform)
- Best model selected automatically based on validation RMSE

---

## 🧪 Experiment Tracking & Model Management
- **MLflow Experiments**
  - Parameters
  - Metrics
  - Model artifacts
- **Nested runs** for hyperparameter sweeps
- **MLflow Model Registry**
  - Versioned models
  - Promotion-ready artifacts

---

## 🚀 Inference & Serving Readiness
- Model loaded using **MLflow PyFunc**
- Serving input validated prior to deployment
- Compatible with:
  - REST API serving
  - Batch inference
  - Cloud ML platforms

---

## ☁️ Deployment Readiness
- MLflow-compatible model format
- Can be packaged into:
  - Docker containers
  - Cloud-native serving endpoints
- Clear separation of training, evaluation, and inference

---

## 🛠 Tech Stack
- Python 3.10
- Keras / TensorFlow
- MLflow
- Hyperopt
- Scikit-learn
- Pandas / NumPy

---

## 📊 Key Outcomes
- Automated experiment comparison
- Best-performing ANN selected and registered
- Fully reproducible ML pipeline
- Production-aligned workflow (training → registry → inference)

---

## 🎯 Skills Demonstrated
- MLOps & ML Engineering
- Experiment tracking & reproducibility
- Hyperparameter optimization
- Model versioning & governance
- Deployment-oriented ML design

---
