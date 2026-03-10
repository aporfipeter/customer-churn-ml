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

    print("\n=== Missing Values Per Column (pandas isna) ===")
    print(df.isna().sum())

    print("\n=== Empty String Counts Per Object Column ===")
    object_columns = df.select_dtypes(include="object").columns
    for column in object_columns:
        empty_count = (df[column] == "").sum()
        whitespace_count = df[column].astype(str).str.strip().eq("").sum()
        print(f"{column}: empty_strings={empty_count}, whitespace_only={whitespace_count}")

    print("\n=== Unique Values in Selected Columns ===")
    columns_to_check = [
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "Contract",
        "Churn",
    ]
    for column in columns_to_check:
        print(f"{column}: {df[column].unique().tolist()}")

    print("\n=== Attempting Numeric Conversion of TotalCharges ===")
    total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
    failed_conversion_count = total_charges_numeric.isna().sum()

    print(f"Rows where TotalCharges could not be converted: {failed_conversion_count}")

    if failed_conversion_count > 0:
        print("\n=== Rows with Invalid TotalCharges ===")
        invalid_rows = df[total_charges_numeric.isna()][
            ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
        ]
        print(invalid_rows.head(10))


if __name__ == "__main__":
    main()