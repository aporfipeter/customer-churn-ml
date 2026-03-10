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
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Conclusion](#conclusion)
8. [Future Work](#future-work)

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

## Current Pipeline

```
data/raw ⭢ load_data.py (inspection) ⭢ preprocess.py ⭢ data/processed/telco_churn_clean.csv
```

## Next Steps

- exploratory data analysis
- feature engineering
- baseline model training
- model evaluation
- prediction API
