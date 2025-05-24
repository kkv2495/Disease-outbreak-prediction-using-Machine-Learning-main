import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import pandas as pd

# Define the parameters for GridSearchCV
parameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

def save_metrics_plot(metrics, title, filename):
    """Save a bar plot of metrics."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('stats', exist_ok=True)
    plt.savefig(f'stats/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def train_model(preprocess_data):
    """
    Train a RandomForest model using GridSearchCV.
    Saves the best model and returns evaluation metrics.
    """
    # Create necessary directories
    os.makedirs('Model', exist_ok=True)
    
    # Preprocess the data
    X_train, X_test, X_oob, y_train, y_test, y_oob, scaler = preprocess_data()
    
    # Initialize the base model
    rf = RandomForestClassifier(random_state=42)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=parameters,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )
    
    # Train the model with GridSearch
    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Generate timestamp for model saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the best model and scaler
    os.makedirs('Saved_Models', exist_ok=True)
    model_path = f'Saved_Models/model_{timestamp}.joblib'
    scaler_path = f'Saved_Models/scaler_{timestamp}.joblib'
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Also save with fixed names for easier access
    joblib.dump(best_model, 'Saved_Models/best_model.joblib')
    joblib.dump(scaler, 'Saved_Models/scaler.joblib')
    
    # Evaluate on test set
    y_pred_test = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Store metrics
    metrics = {
        'test_accuracy': test_accuracy,
        'best_params': grid_search.best_params_,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'timestamp': timestamp
    }
    
    # Save metrics to a file
    with open('Saved_Models/metrics.json', 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    
    print("\nTraining completed!")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Model saved to: {model_path}")
    
    return best_model, metrics


if __name__ == "__main__":
    from preprocess import preprocess_data
    model, metrics = train_model(preprocess_data)