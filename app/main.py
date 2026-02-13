from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict_transaction

app = FastAPI(title="Fraud Detection API")

class Transaction(BaseModel):
    features: list[float]


class PredictionResponse(BaseModel):
    fraud_prediction: int
    fraud_probability: float

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: Transaction):
    if len(data.features) != 29:
        raise HTTPException(
            status_code=400,
            detail="Expected exactly 29 numeric features in trained order.",
        )

    try:
        prediction, probability = predict_transaction(data.features)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error.",
        ) from exc

    return PredictionResponse(
        fraud_prediction=prediction,
        fraud_probability=probability,
    )