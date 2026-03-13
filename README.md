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
6. [Model Training](#model-training)
7. [Current Pipeline](#current-pipeline)
8. [Next Steps](#next-steps)

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

## Model Training

Model training is performed using `train.py`.

### Method

- Logistic Regression: choosing `max_iter=1000` to ensure convergence.
- `model.predict` returns class labels (0 or 1)
- `model.predict_proba` returns probabilities for each class.

### Evaluation Metrics

- Accuracy
- ROC-AUC
- Confusion Matrix
- Classification Report

### Baseline results

Using Logistic Regression with default parameters (probability threshold = 0.5)
Accuracy: 0.80
ROC-AUC: 0.84

#### Confusion Matrix

```
[[926 109]
 [166 208]]
```

#### Classification Report

```
              precision    recall  f1-score   support

           0       0.85      0.89      0.87      1035
           1       0.66      0.56      0.60       374

    accuracy                           0.80      1409
   macro avg       0.75      0.73      0.74      1409
weighted avg       0.80      0.80      0.80      1409
```

#### Feature Importance

```
=== Top 10 Features Increasing Churn Risk ===
InternetService_Fiber optic       0.742498
PaymentMethod_Electronic check    0.380147
PaperlessBilling_Yes              0.376937
MultipleLines_Yes                 0.276279
StreamingTV_Yes                   0.198993
StreamingMovies_Yes               0.198248
MultipleLines_No phone service    0.167618
SeniorCitizen                     0.147555
PaymentMethod_Mailed check        0.070644
Partner_Yes                       0.023491
dtype: float64

=== Top 10 Features Decreasing Churn Risk ===
TechSupport_No internet service       -0.110744
StreamingTV_No internet service       -0.110744
StreamingMovies_No internet service   -0.110744
OnlineBackup_Yes                      -0.186234
Dependents_Yes                        -0.223782
TechSupport_Yes                       -0.388555
OnlineSecurity_Yes                    -0.436367
PhoneService_Yes                      -0.438808
Contract_One year                     -0.677463
Contract_Two year                     -1.303864
dtype: float64
```

Note: `max_iter` limit reached. Scaling and increasing iterations may improve results.

### Results after feature scaling

```
=== Accuracy ===
0.8048261178140526

=== ROC-AUC ===
0.8426903304141156
```

#### Confusion Matrix

```
[[926 109]
 [166 208]]
```

#### Classification Report

```
              precision    recall  f1-score   support

           0       0.85      0.89      0.87      1035
           1       0.66      0.57      0.61       374

    accuracy                           0.81      1409
   macro avg       0.75      0.73      0.74      1409
weighted avg       0.80      0.81      0.80      1409
```

#### Feature Importance

```
=== Top 10 Features Increasing Churn Risk ===
InternetService_Fiber optic       0.776154
TotalCharges                      0.514285
StreamingMovies_Yes               0.257227
StreamingTV_Yes                   0.257144
MultipleLines_Yes                 0.216167
PaperlessBilling_Yes              0.182034
PaymentMethod_Electronic check    0.181103
SeniorCitizen                     0.053073
DeviceProtection_Yes              0.053048
PaymentMethod_Mailed check        0.032262
dtype: float64

=== Top 10 Features Decreasing Churn Risk ===
StreamingMovies_No internet service   -0.092761
OnlineSecurity_No internet service    -0.092761
StreamingTV_No internet service       -0.092761
TechSupport_Yes                       -0.100502
Dependents_Yes                        -0.103550
OnlineSecurity_Yes                    -0.123561
Contract_One year                     -0.285509
Contract_Two year                     -0.586859
MonthlyCharges                        -0.920153
tenure                                -1.236528
dtype: float64
```

### Random Forest results

```
=== Accuracy ===
0.7955997161107168

=== ROC-AUC ===
0.8266746751401483
```

#### Classification Report

```
=== Classification Report ===
              precision    recall  f1-score   support

           0       0.84      0.90      0.87      1035
           1       0.64      0.51      0.57       374

    accuracy                           0.80      1409
   macro avg       0.74      0.71      0.72      1409
weighted avg       0.79      0.80      0.79      1409
```

#### Confusion Matrix

```
[[929 106]
 [182 192]]
```

#### Feature Importance

```
TotalCharges                      0.193091
tenure                            0.168945
MonthlyCharges                    0.168473
InternetService_Fiber optic       0.038828
PaymentMethod_Electronic check    0.038669
Contract_Two year                 0.031733
gender_Male                       0.028549
OnlineSecurity_Yes                0.027506
PaperlessBilling_Yes              0.025543
Partner_Yes                       0.023383
dtype: float64
```

### Conclusion

The logistic regression model achieved an accuracy of 80% and an ROC-AUC score of 84%. The random forest model achieved an accuracy of 79.6% and an ROC-AUC score of 82.7%. Logistic regression did slightly better.
Potential reasons:

- Strong linear signals
- Small feature space, relatively simple structure
- Random Forest was used with default parameters. Further tuning could probably improve the model.

## Current Pipeline

```

data/raw ⭢ load_data.py (inspection) ⭢ preprocess.py ⭢ data/processed/telco_churn_clean.csv ⭢ EDA ⭢ feature engineering ⭢ train / test split ⭢ scaled pipeline ⭢ model training using logistic regression ⭢ model training using random forest ⭢ model evaluation

```

## Next Steps

- exploring different models (eg. random forest)
- prediction API
