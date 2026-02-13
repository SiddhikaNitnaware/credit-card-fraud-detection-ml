import joblib
import numpy as np

model_package = joblib.load("model/fraud_model.pkl")

model = model_package["model"]
scaler = model_package["scaler"]
threshold = model_package["threshold"]

def predict_transaction(features):
    features = np.array(features).reshape(1, -1)
    features[:, -1] = scaler.transform(features[:, -1].reshape(-1, 1)).flatten()

    prob = model.predict_proba(features)[:, 1][0]
    prediction = int(prob >= threshold)

    return prediction, prob