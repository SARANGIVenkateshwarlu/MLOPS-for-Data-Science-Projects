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
# Project -4:  Machine Learning Pipeline with DVC & MLflow
📌 Project Overview

This project demonstrates how to build an end-to-end machine learning pipeline using DVC (Data Version Control) for data and model versioning and MLflow for experiment tracking.
The pipeline trains and evaluates a Random Forest Classifier on the Pima Indians Diabetes Dataset, following best practices for reproducibility and MLOps.

The goal of this project is to show how data, code, models, and experiments can be tracked together in a structured and scalable way. (dagshub.com)
🚀 Key Features

    ✅ End-to-end ML pipeline
    ✅ Data and model versioning with DVC
    ✅ Experiment tracking with MLflow
    ✅ Reproducible pipeline stages
    ✅ Modular and scalable project structure
    ✅ Integration-ready with cloud storage (S3 / GCS / Azure)

🧰 Tech Stack

    Python
    Scikit-learn
    DVC – data & pipeline versioning
    MLflow – experiment tracking
    Git / DagsHub – code, data, and collaboration

📂 Project Structure
text

machinelearningpipeline/
│
├── data/                  # Raw and processed datasets (DVC tracked)
├── src/                   # Source code for pipeline stages
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── models/                # Trained models (DVC tracked)
├── dvc.yaml               # DVC pipeline definition
├── params.yaml            # Model and pipeline parameters
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

🔄 Pipeline Stages

    Data Preprocessing
        Cleans and prepares the dataset
        Splits data into training and test sets

    Model Training
        Trains a Random Forest Classifier
        Logs parameters and metrics to MLflow

    Model Evaluation
        Evaluates model performance
        Tracks evaluation metrics for comparison

▶️ How to Run the Project
1️⃣ Clone the Repository
bash

git clone https://dagshub.com/venky.sarangi/machinelearningpipeline
cd machinelearningpipeline

2️⃣ Create Environment & Install Dependencies
bash

pip install -r requirements.txt

3️⃣ Run the DVC Pipeline
bash

dvc repro

📊 Experiment Tracking

All experiments, metrics, and model artifacts are logged using MLflow.
You can compare runs and analyze performance directly from the DagsHub UI.
☁️ Data & Model Versioning

    Datasets and models are tracked using DVC
    Supports remote storage (AWS S3, GCS, Azure, or DagsHub Storage)
    Ensures full reproducibility across environments

🎯 Use Cases

    Learning MLOps fundamentals
    Building reproducible ML workflows
    Experiment tracking and model comparison
    Production-ready ML pipeline templates

🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests to improve the pipeline.
📜 License

This project is intended for educational and learning purposes.
⭐ If you find this project useful, consider starring it!

If you want, I can also:

    Tailor this README for GitHub vs DagsHub
    Add badges (Python, DVC, MLflow)
    Create a Beginner-friendly version
    Generate a project architecture diagram


# Project -5  🐳 Basic Hello Docker Project 

- `docker --version` → Verify Docker installation  
- `docker build -t welcome-app .` → Build Docker image  
- `docker run -p 5000:5000 welcome-app` → Run the container  
- `docker ps` → Check running containers
