import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess_data(oob_size=0.15, test_size=0.2, random_state=42):
    """
    Preprocess the data and split into train, test, and OOB sets.
    
    Args:
        oob_size: Proportion of data to use for out-of-bag evaluation
        test_size: Proportion of remaining data to use for test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, X_oob, y_train, y_test, y_oob, scaler
    """
    # Load dataset (replace with your actual data loading code)
    df = pd.read_csv('data/diabetes.csv')  # Update with your dataset path
    
   # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs('Saved_Models', exist_ok=True)
    joblib.dump(scaler, 'Saved_Models/scaler_diabetes.sav')
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess_data()