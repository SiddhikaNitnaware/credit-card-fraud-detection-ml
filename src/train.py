from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

from preprocessing import load_and_sample_data

X, y, scaler = load_and_sample_data("data/creditcard.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=40,
    max_depth=8,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train_res, y_train_res)

model_package = {
    "model": rf,
    "scaler": scaler,
    "threshold": 0.4
}

joblib.dump(model_package, "model/fraud_model.pkl")