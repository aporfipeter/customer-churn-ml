from pathlib import Path

import pandas as pd


def load_raw_data() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = pd.read_csv(data_path)

    return df


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TotalCharges to numeric and handle invalid rows.
    """

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    missing_count = df["TotalCharges"].isna().sum()
    print(f"\nRows with missing TotalCharges after conversion: {missing_count}")

    if missing_count > 0:
        print("Filling missing TotalCharges with 0 (tenure=0 customers)")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


def save_processed_data(df: pd.DataFrame) -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "processed" / "telco_churn_clean.csv"

    df.to_csv(output_path, index=False)

    print(f"\nProcessed dataset saved to: {output_path}")


def main() -> None:
    print("\nLoading raw dataset...")
    df = load_raw_data()

    print("\nCleaning TotalCharges column...")
    df = clean_total_charges(df)

    print("\nFinal data types:")
    print(df.dtypes)

    print("\nSaving cleaned dataset...")
    save_processed_data(df)


if __name__ == "__main__":
    main()