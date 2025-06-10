import joblib
import numpy as np

def load_diabetes_model_and_scaler():
    """Load the diabetes model and scaler"""
    try:
        model = joblib.load('Saved_Models/diabetes_model.joblib')
        scaler = joblib.load('Saved_Models/scaler_diabetes.joblib')
        return model, scaler
    except Exception as e:
        raise Exception(f"Error loading model or scaler: {str(e)}")

def predict_diabetes(features):
    """Make prediction for diabetes"""
    model, scaler = load_diabetes_model_and_scaler()
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)
    return prediction[0], probability[0][1]  # Return prediction and probability