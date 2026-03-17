from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Customer Churn Prediction API")


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def load_model():
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "artifacts" / "churn_logistic_pipeline.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)
    return model


model = load_model()


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    input_df = pd.DataFrame([customer.model_dump()])

    churn_probability = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    return {
        "churn_probability": round(float(churn_probability), 4),
        "prediction": int(prediction),
        "prediction_label": "churn" if prediction == 1 else "no churn",
    }