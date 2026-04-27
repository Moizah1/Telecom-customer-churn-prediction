# 📉 Telco Customer Churn Prediction

A machine learning project to predict customer churn for a telecom company using XGBoost, with an interactive Streamlit dashboard for real-time predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Features](#features)

---

## Overview

Customer churn — when customers stop using a service — is a critical business problem in the telecom industry. This project builds a binary classification model to predict whether a customer will churn, enabling proactive retention strategies.

Key highlights:
- **Dataset**: 7,043 telecom customers with 20 features
- **Model**: XGBoost classifier with SMOTE oversampling to handle class imbalance (~2.77:1 ratio)
- **Interface**: Interactive Streamlit web app for real-time churn prediction

---

## Dataset

The dataset (`TelcoCustomerChurn.csv`) contains customer information across 21 columns:

| Category | Features |
|---|---|
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Target** | `Churn` (Yes / No) |

**Class Distribution**: ~73.5% No Churn vs ~26.5% Churn (2.77:1 imbalance, handled with SMOTE)

---

## Project Structure

```
telco-churn/
│
├── TelcoCustomerChurn.csv     # Raw dataset
├── customer_churn.ipynb       # EDA, preprocessing & model training notebook
├── churn_app.py               # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd telco-churn
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run the Notebook

Open `customer_churn.ipynb` in Jupyter to explore the data, train the model, and generate the saved model file:

```bash
jupyter notebook customer_churn.ipynb
```

### Launch the Streamlit App

```bash
streamlit run churn_app.py
```

Then open your browser at `http://localhost:8501`.

---

## Model Details

| Component | Choice |
|---|---|
| **Algorithm** | XGBoost Classifier |
| **Imbalance Handling** | SMOTE (Synthetic Minority Oversampling) |
| **Encoding** | Label Encoding for categorical features |
| **Train/Test Split** | Standard split via `train_test_split` |
| **Evaluation** | Accuracy, Confusion Matrix, Classification Report |

### Preprocessing Steps
1. Drop `customerID` (non-predictive identifier)
2. Convert `TotalCharges` from string to float (handle blank entries as `0.0`)
3. Label-encode all categorical columns
4. Apply SMOTE to balance the training set

---

## Features

- **Exploratory Data Analysis** — Distribution plots for `tenure`, `MonthlyCharges`, and `TotalCharges`
- **Class Imbalance Detection** — Quantified and mitigated with SMOTE
- **XGBoost Model** — Fast, high-performance gradient boosting
- **Streamlit Dashboard** — Interactive UI built with Plotly for visualization

---

## Dependencies

```
numpy, pandas, matplotlib, seaborn
scikit-learn, imbalanced-learn, xgboost
streamlit, plotly, joblib
```

Install all with:
```bash
pip install -r requirements.txt
```
