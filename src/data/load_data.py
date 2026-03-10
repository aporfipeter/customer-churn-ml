from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    print("\n=== Dataset Loaded Successfully ===")
    print(f"Path: {data_path}")

    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Columns ===")
    print(df.columns.tolist())

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== First 5 Rows ===")
    print(df.head())

    print("\n=== Missing Values Per Column ===")
    print(df.isna().sum())


if __name__ == "__main__":
    main()