import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_parkinsons_data():
    # Load data
    df = pd.read_csv('data/parkinsons.csv')
    
    # Separate features and target
    X = df.drop(['name', 'status'], axis=1)
    y = df['status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs('Saved_Models', exist_ok=True)
    joblib.dump(scaler, 'Saved_Models/scaler_parkinsons.joblib')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess_parkinsons_data()