## Credit Card Fraud Detection ML System
An end-to-end machine learning system that detects fraudulent credit card transactions and exposes real-time predictions through a FastAPI backend with a Streamlit user interface.

# Features
-Handles extreme class imbalance using SMOTE
-Trained Random Forest Classifier with optimized hyperparameters
-Threshold tuning to balance fraud detection vs false alerts
-REST API deployment using FastAPI
-Interactive frontend using Streamlit

# Model Performance
-ROC-AUC Score: ~0.97
-Fraud Recall: ~0.89
-Precision optimized via threshold tuning (0.4)

# Setup Instructions:
1. Clone Repository
git clone https://github.com/SiddhikaNitnaware/credit-card-fraud-detection-ml.git
2. Create Virtual Environment
>>python -m venv .venv
>>.venv\Scripts\activate
3. install Dependencies
pip install -r requirements.txt
4. Download Dataset
Download from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
5. Train Model
python src/train.py
6. Run API
python -m uvicorn app.main:app --reload
7. Run Streamlit UI (in new terminal)
streamlit run app/streamlit_app.py