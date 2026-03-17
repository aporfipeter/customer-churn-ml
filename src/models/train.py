from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_processed_data() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "telco_churn_clean.csv"

    df = pd.read_csv(data_path)
    return df

def prepare_train_test_data(df: pd.DataFrame):
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn", "customerID"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    categorical_columns = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_columns = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_columns),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ])

    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Accuracy ===")
    print(accuracy_score(y_test, y_pred))

    print("\n=== ROC-AUC ===")
    print(roc_auc_score(y_test, y_prob))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))


def show_feature_importance(model, X_train):
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importance = pd.Series(classifier.coef_[0], index=feature_names)
    importance = importance.sort_values(ascending=False)

    print("\n=== Top 10 Features Increasing Churn Risk ===")
    print(importance.head(10))

    print("\n=== Top 10 Features Decreasing Churn Risk ===")
    print(importance.tail(10))

def show_feature_importance_rf(model, X_train):

    importance = pd.Series(model.feature_importances_, index=X_train.columns)

    importance = importance.sort_values(ascending=False)

    print("\n=== Random Forest Top Features ===")
    print(importance.head(10))

def analyze_thresholds(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    print("\n=== Threshold Analysis (sample points) ===")

    for i in np.linspace(0, len(thresholds)-1, 10, dtype=int):
        print(
            f"Threshold: {thresholds[i]:.2f} | "
            f"Precision: {precision[i]:.2f} | "
            f"Recall: {recall[i]:.2f}"
        )

def save_model(model):
    project_root = Path(__file__).resolve().parents[2]
    artifacts_path = project_root / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_path / "churn_logistic_pipeline.pkl"
    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")

def main():
    print("\nLoading processed dataset...")
    df = load_processed_data()

    print("\nPreparing raw train/test split...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(df)

    print("\nTraining Logistic Regression pipeline on raw inputs...")
    model = train_model(X_train, y_train)

    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)

    show_feature_importance(model, X_train)

    analyze_thresholds(model, X_test, y_test)

    print("\nSaving Logistic Regression pipeline...")
    save_model(model)
    
    """
    print("\nTraining Random Forest model...")
    rf_model = train_random_forest(
        pd.get_dummies(X_train, drop_first=True),
        y_train
    )

    print("\nEvaluating Random Forest...")
    evaluate_model(rf_model, pd.get_dummies(X_test, drop_first=True), y_test)

    show_rf_importance(rf_model, pd.get_dummies(X_train, drop_first=True))
    """


if __name__ == "__main__":
    main()