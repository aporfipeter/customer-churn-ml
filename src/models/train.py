from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def load_datasets():
    project_root = Path(__file__).resolve().parents[2]
    processed_path = project_root / "data" / "processed"

    X_train = pd.read_csv(processed_path / "X_train.csv")
    X_test = pd.read_csv(processed_path / "X_test.csv")
    y_train = pd.read_csv(processed_path / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(processed_path / "y_test.csv").squeeze("columns")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
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

    importance = pd.Series(model.coef_[0], index=X_train.columns)

    importance = importance.sort_values(ascending=False)

    print("\n=== Top 10 Features Increasing Churn Risk ===")
    print(importance.head(10))

    print("\n=== Top 10 Features Decreasing Churn Risk ===")
    print(importance.tail(10))


def main():
    print("\nLoading train/test datasets...")
    X_train, X_test, y_train, y_test = load_datasets()

    print("\nTraining Logistic Regression model...")
    model = train_model(X_train, y_train)

    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)

    print("\nFeature Importance")
    show_feature_importance(model, X_train)


if __name__ == "__main__":
    main()