from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_processed_data() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "telco_churn_clean.csv"

    df = pd.read_csv(data_path)

    return df


def separate_features_and_target(df: pd.DataFrame):
    """
    Separate the target variable from the feature matrix.
    """

    y = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Churn", "customerID"])

    return X, y


def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical variables to numeric using one-hot encoding.
    """

    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded


def split_dataset(X, y):
    """
    Split dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def save_datasets(X_train, X_test, y_train, y_test):

    project_root = Path(__file__).resolve().parents[2]
    processed_path = project_root / "data" / "processed"

    X_train.to_csv(processed_path / "X_train.csv", index=False)
    X_test.to_csv(processed_path / "X_test.csv", index=False)
    y_train.to_csv(processed_path / "y_train.csv", index=False)
    y_test.to_csv(processed_path / "y_test.csv", index=False)

    print("\nDatasets saved successfully.")


def main():

    print("\nLoading processed dataset...")
    df = load_processed_data()

    print("\nSeparating features and target...")
    X, y = separate_features_and_target(df)

    print("\nEncoding categorical features...")
    X_encoded = encode_categorical_features(X)

    print("\nFeature matrix shape:")
    print(X_encoded.shape)

    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X_encoded, y)

    print("\nTraining set size:", X_train.shape)
    print("Test set size:", X_test.shape)

    print("\nSaving datasets...")
    save_datasets(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()