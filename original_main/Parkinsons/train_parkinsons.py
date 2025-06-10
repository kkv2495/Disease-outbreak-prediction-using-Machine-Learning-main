import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess_parkinsons import preprocess_parkinsons_data

def train_parkinsons_model():
    # Ensure directories exist
    os.makedirs('Images/Parkinsons', exist_ok=True)
    os.makedirs('Saved_Models', exist_ok=True)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_parkinsons_data()
    
    # After training, save the feature names
    joblib.dump(list(X_train.columns), 'Saved_Models/feature_names_parkinsons.joblib')
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )
    
    print("Starting Grid Search for Parkinson's Disease Model...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("\nBest Parameters:", grid_search.best_params_)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save model
    joblib.dump(best_model, 'Saved_Models/parkinsons_model.joblib')
    
    # Save classification report
    report = classification_report(y_test, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    os.makedirs('Images/Parkinsons', exist_ok=True)
    report_df.to_csv('Images/Parkinsons/classification_report.csv')
    
    return best_model, scaler

if __name__ == "__main__":
    model, scaler = train_parkinsons_model()