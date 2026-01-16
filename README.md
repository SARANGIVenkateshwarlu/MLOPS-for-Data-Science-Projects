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

## ✅ Completed Steps

### 1️⃣ Environment Setup
- Created a Python virtual environment in **VS Code**
- Activated the environment for isolated dependency management

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
