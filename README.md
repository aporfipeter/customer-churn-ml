# Project: Customer Churn Prediction System

## Goal

This project aims to develop a machine learning model to predict customer churn for a telecom company. Customer churn is a critical business problem as it directly impacts revenue and customer satisfaction. By accurately predicting which customers are likely to churn, the company can take proactive measures to retain them.

The project includes:

- Data ingestion
- Data exploration and preprocessing
- Feature engineering
- Model training and evaluation
- Prediction API

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Cleaning Decisions](#data-cleaning-decisions)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Engineering](#feature-engineering)
6. [Current Pipeline](#current-pipeline)
7. [Next Steps](#next-steps)

## Introduction

Customer churn prediction involves analyzing customer data to identify patterns and factors that contribute to churn. The goal is to build a predictive model that can classify customers as either "churned" or "not churned" based on their characteristics.

## Dataset

The dataset used: Kaggle - Telco Customer Churn

### Dataset Overview

```bash
Rows: 7043
Columns: 21
Target: Churn (Yes/No)
```

Key features include:

- tenure
- contract type
- monthly charges
- internet services
- technical support
- payment method

## Data Cleaning Decisions

Issue discovered:
`TotalCharges` column was stored as string due to blank values.

Investigation showed:

```bash
tenure = 0
TotalCharges = blank
```

This represents customers who have just joined and have not yet accumulated charges.

Cleaning decision:

```python
TotalCharges = 0 for tenure = 0 rows
```

Implementation:

```python
pd.to_numeric(errors="coerce")
fillna(0)
```

## Exploratory Data Analysis

Exploratory Data Analysis (EDA)is performed using Jupyter Notebooks.

`eda_telco_churn.pynb` examines the dataset by describing the data (shape, data types, missing values, descriptive statistics). Churn distribution is examined within categories of internet service and contract type. The effect of monthly charges and tenure on churn is also examined.

## Feature Engineering

Feature engineering is performed using `build_features.py`.

- Convert target variable `churn` to binary (0/1)
- Remove CustomerId to avoid explicit memorization of customers.
- One-hot encode categorical variables. `drop_first=True` is used to avoid multicollinearity.
- Train / Test split: test data set size: 20%; Using `stratify=y` to ensure churn distribution is preserved in both sets.
- Generated input files for model training.

### Input files for model training

- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`

## Current Pipeline

```
data/raw ⭢ load_data.py (inspection) ⭢ preprocess.py ⭢ data/processed/telco_churn_clean.csv ⭢ EDA ⭢ feature matrix ⭢ train / test split ⭢ model training (NEXT STEP)
```

## Next Steps

- baseline model training
- model evaluation
- prediction API
