from fastapi import FastAPI
import pandas as pd
from src.data_loader import load_data
from src.predict import predict_anomaly

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Autoencoder Anomaly Detection API"}

@app.get("/predict")
def predict():
    X, _ = load_data("data/creditcard.csv")
    result = predict_anomaly(X[:100])  # small sample
    return {"anomalies": result}