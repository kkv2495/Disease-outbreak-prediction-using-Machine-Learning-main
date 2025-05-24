import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import subprocess
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# Set page config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Set paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "Saved_Models"
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "Images"

# Create necessary directories
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Sidebar for navigation
st.sidebar.title("Disease Prediction System")
page = st.sidebar.radio("Navigation", ["Predict", "Evaluate Models"])

def get_model_info(disease_name):
    """Get model information including last modified time and size"""
    model_name = disease_name.lower().replace(' ', '_').replace("'", '')
    model_path = MODELS_DIR / f"{model_name}_model.joblib"
    
    if not model_path.exists():
        return None
        
    stats = model_path.stat()
    return {
        'name': disease_name,
        'path': str(model_path),
        'size_mb': stats.st_size / (1024 * 1024),
        'modified': pd.Timestamp(stats.st_mtime, unit='s')
    }

def load_model(disease_name):
    """Load model and scaler for the specified disease"""
    try:
        # Map disease names to their corresponding model filenames
        model_name_map = {
            'diabetes': 'diabetes',
            'heart': 'heart',  # This will use heart_model.joblib and scaler_heart.joblib
            'parkinsons': 'parkinsons'
        }
        
        # Get the base model name
        base_name = disease_name.lower().replace(' ', '_').replace("'", '')
        model_name = model_name_map.get(base_name, base_name)
        
        # Special case for heart disease to use the correct model filename
        if 'heart' in base_name:
            model_path = MODELS_DIR / "heart_disease_model.joblib"
            scaler_path = MODELS_DIR / "scaler_heart.joblib"
        else:
            model_path = MODELS_DIR / f"{model_name}_model.joblib"
            scaler_path = MODELS_DIR / f"scaler_{model_name}.joblib"
        
        if not model_path.exists() or not scaler_path.exists():
            st.warning(f"Model files not found. Looking for:\n- {model_path.name}\n- {scaler_path.name}")
            return None, None, None
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load feature names if available
        features_path = MODELS_DIR / f"feature_names_{model_name}.joblib"
        feature_names = joblib.load(features_path) if features_path.exists() else None
        
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def plot_metrics(y_true, y_pred, y_prob, disease_name):
    """Generate evaluation metrics and plots"""
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"])
    
    # Confusion Matrix
    with tab1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {disease_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(plt)
        plt.close()
    
    # ROC Curve
    roc_auc = 0
    with tab2:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {disease_name}')
            plt.legend(loc="lower right")
            st.pyplot(plt)
            plt.close()
        except Exception as e:
            st.error(f"Error plotting ROC curve: {str(e)}")
    
    # Precision-Recall Curve
    avg_precision = 0
    with tab3:
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            avg_precision = average_precision_score(y_true, y_prob[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, lw=2, 
                    label=f'AP = {avg_precision:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {disease_name}')
            plt.legend(loc="lower left")
            st.pyplot(plt)
            plt.close()
        except Exception as e:
            st.error(f"Error plotting Precision-Recall curve: {str(e)}")
    
    # Display metrics
    st.subheader("Classification Report")
    st.json(report)
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }

# Prediction Page
if page == "Predict":
    st.title("Disease Prediction")
    
    disease = st.sidebar.selectbox(
        "Select Disease for Prediction",
        ["Diabetes", "Heart_Disease", "Parkinsons"]
    )
    
    # Show model info in sidebar
    model_info = get_model_info(disease)
    if model_info:
        st.sidebar.subheader("Model Information")
        st.sidebar.write(f"**Name:** {model_info['name']}")
        st.sidebar.write(f"**Size:** {model_info['size_mb']:.2f} MB")
        st.sidebar.write(f"**Last Modified:** {model_info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.sidebar.warning("No trained model found. Please train the model first.")
    
    # Train button
    if st.sidebar.button(f"Train {disease} Model"):
        with st.spinner(f"Training {disease} model. This may take a few minutes..."):
            try:
                if disease == "Diabetes":
                    script_path = str(BASE_DIR / "diabetes" / "train_diabetes.py")
                elif disease == "Heart_Disease":
                    script_path = str(BASE_DIR / "heart" / "train_heart.py")
                else:  # Parkinson's
                    script_path = str(BASE_DIR / "Parkinsons" / "train_parkinsons.py")
                
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    st.sidebar.success(f"{disease} model trained successfully!")
                    st.sidebar.code(result.stdout)
                    # Refresh model info
                    model_info = get_model_info(disease)
                else:
                    st.sidebar.error(f"Training failed: {result.stderr}")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    # Input fields and prediction
    model, scaler, feature_names = load_model(disease)
    
    if disease == "Diabetes":
        st.header("Diabetes Prediction")
        st.write("Enter patient information to predict diabetes risk.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input('Pregnancies', 0, 17, 3)
            glucose = st.number_input('Glucose', 0, 200, 117)
            blood_pressure = st.number_input('Blood Pressure', 0, 122, 72)
            skin_thickness = st.number_input('Skin Thickness (mm)', 0, 99, 23)
            
        with col2:
            insulin = st.number_input('Insulin (mu U/ml)', 0, 846, 30)
            bmi = st.number_input('BMI', 0.0, 67.1, 32.0)
            diabetes_pedigree = st.number_input('Diabetes Pedigree Function', 0.08, 2.42, 0.37)
            age = st.number_input('Age', 21, 81, 29)
        
        if st.button('Predict Diabetes'):
            if model is None or scaler is None:
                st.warning("Model not found. Please train the model first using the sidebar button.")
            else:
                try:
                    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                         insulin, bmi, diabetes_pedigree, age]])
                    input_df = pd.DataFrame(input_data, columns=[
                        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
                    ])
                    
                    input_scaled = scaler.transform(input_df)
                    prediction = model.predict(input_scaled)
                    probability = model.predict_proba(input_scaled)
                    
                    st.subheader('Prediction Result')
                    if prediction[0] == 1:
                        st.error(f"High risk of diabetes (Probability: {probability[0][1]:.2%})")
                    else:
                        st.success(f"Low risk of diabetes (Probability: {probability[0][0]:.2%})")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    # Load model and scaler
    model, scaler, feature_names = load_model(disease)
    
    if disease == "Heart_Disease":
        st.header("Heart_Disease Prediction")
        st.write("Enter patient information to predict Heart_Disease risk.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age', 29, 77, 54)
            sex = st.selectbox('Sex', ['Male', 'Female'])
            cp = st.selectbox('Chest Pain Type', 
                            ['Typical Angina', 'Atypical Angina', 
                             'Non-anginal Pain', 'Asymptomatic'])
            trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 94, 200, 131)
            chol = st.number_input('Serum Cholestoral (mg/dl)', 126, 564, 246)
            
        with col2:
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
            restecg = st.selectbox('Resting Electrocardiographic Results', 
                                  ['Normal', 'ST-T Wave Abnormality', 
                                   'Left Ventricular Hypertrophy'])
            thalach = st.number_input('Maximum Heart Rate Achieved', 71, 202, 150)
            exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
            oldpeak = st.number_input('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
            slope = st.selectbox('Slope of the Peak Exercise ST Segment', 
                               ['Upsloping', 'Flat', 'Downsloping'])
            ca = st.number_input('Number of Major Vessels Colored by Flourosopy', 0, 4, 1)
            thal = st.selectbox('Thalassemia', 
                              ['Normal', 'Fixed Defect', 'Reversible Defect'])
        
        if st.button('Predict Heart_Disease'):
            if model is None or scaler is None:
                st.warning("Model not found. Please train the model first using the sidebar button.")
            else:
                try:
                    # Process inputs
                    sex = 1 if sex == 'Male' else 0
                    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 
                             'Non-anginal Pain': 2, 'Asymptomatic': 3}
                    cp = cp_map[cp]
                    fbs = 1 if fbs == 'Yes' else 0
                    restecg_map = {'Normal': 0, 'ST-T Wave Abnormality': 1, 
                                  'Left Ventricular Hypertrophy': 2}
                    restecg = restecg_map[restecg]
                    exang = 1 if exang == 'Yes' else 0
                    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
                    slope = slope_map[slope]
                    thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
                    thal = thal_map[thal]
                    
                    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                          thalach, exang, oldpeak, slope, ca, thal]])
                    input_df = pd.DataFrame(input_data, columns=[
                        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
                    ])
                    
                    input_scaled = scaler.transform(input_df)
                    prediction = model.predict(input_scaled)
                    probability = model.predict_proba(input_scaled)
                    
                    st.subheader('Prediction Result')
                    if prediction[0] == 1:
                        st.error(f"High risk of Heart_Disease (Probability: {probability[0][1]:.2%})")
                    else:
                        st.success(f"Low risk of Heart_Disease (Probability: {probability[0][0]:.2%})")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    elif disease == "Parkinsons":
        st.header("Parkinsons Prediction")
        st.write("Enter voice measurement data to predict Parkinsons risk.")
        
        model, scaler, feature_names = load_model(disease)
        col1, col2 = st.columns(2)
        
        with col1:
            mdvp_fo = st.number_input('MDVP:Fo(Hz)', 88.33, 260.105, 148.79)
            mdvp_fhi = st.number_input('MDVP:Fhi(Hz)', 102.15, 592.03, 163.78)
            mdvp_flo = st.number_input('MDVP:Flo(Hz)', 65.48, 239.17, 104.18)
            mdvp_jitter = st.number_input('MDVP:Jitter(%)', 0.00168, 0.03316, 0.00484)
            mdvp_jitter_abs = st.number_input('MDVP:Jitter(Abs)', 0.000007, 0.00026, 0.00003)
            mdvp_rap = st.number_input('MDVP:RAP', 0.00068, 0.02144, 0.00218)
            mdvp_ppq = st.number_input('MDVP:PPQ', 0.00092, 0.01958, 0.00236)
            jitter_ddp = st.number_input('Jitter:DDP', 0.00204, 0.06433, 0.00654)
        
        with col2:
            mdvp_shimmer = st.number_input('MDVP:Shimmer', 0.00954, 0.11908, 0.02337)
            mdvp_shimmer_db = st.number_input('MDVP:Shimmer(dB)', 0.085, 1.302, 0.25)
            shimmer_apq3 = st.number_input('Shimmer:APQ3', 0.00408, 0.05647, 0.01184)
            shimmer_apq5 = st.number_input('Shimmer:APQ5', 0.0057, 0.0794, 0.0139)
            mdvp_apq = st.number_input('MDVP:APQ', 0.00719, 0.13778, 0.01723)
            shimmer_dda = st.number_input('Shimmer:DDA', 0.01364, 0.16942, 0.03552)
            nhr = st.number_input('NHR', 0.00065, 0.31482, 0.00751)
            hnr = st.number_input('HNR', 8.441, 33.047, 22.67)
            rpde = st.number_input('RPDE', 0.25657, 0.685151, 0.45978)
            dfa = st.number_input('DFA', 0.574282, 0.825288, 0.71223)
            spread1 = st.number_input('spread1', -7.964984, -2.434031, -5.72)
            spread2 = st.number_input('spread2', 0.006274, 0.450493, 0.1776)
            d2 = st.number_input('D2', 1.423287, 3.671155, 2.44685)
            ppe = st.number_input('PPE', 0.044539, 0.527367, 0.1256)
        
        if st.button("Predict Parkinsons"):
            if model is None or scaler is None:
                st.warning("Model not found. Please train the model first using the sidebar button.")
            else:
                try:
                    input_data = np.array([[
                        mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, mdvp_jitter_abs, 
                        mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db,
                        shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr,
                        rpde, dfa, spread1, spread2, d2, ppe
                    ]])
                    
                    # Create DataFrame with feature names
                    feature_columns = [
                        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 
                        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 
                        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 
                        'spread1', 'spread2', 'D2', 'PPE'
                    ]
                    input_df = pd.DataFrame(input_data, columns=feature_columns)
                    
                    input_scaled = scaler.transform(input_df)
                    prediction = model.predict(input_scaled)
                    probability = model.predict_proba(input_scaled)
                    
                    st.subheader('Prediction Result')
                    if prediction[0] == 1:
                        st.error(f"High risk of Parkinson's disease (Probability: {probability[0][1]:.2%})")
                    else:
                        st.success(f"Low risk of Parkinson's disease (Probability: {probability[0][0]:.2%})")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

# Model Evaluation Page
else:
    st.title("Model Evaluation")
    
    disease = st.selectbox(
        "Select Disease Model to Evaluate",
        ["Diabetes", "Heart_Disease", "Parkinsons"]
    )
    
    model, scaler, feature_names = load_model(disease)
    
    if model is None or scaler is None:
        st.warning(f"No trained model found for {disease}. Please train the model first.")
    else:
        # Load test data
        try:
            if disease == "Diabetes":
                df = pd.read_csv(DATA_DIR / "diabetes.csv")
                X = df.drop('Outcome', axis=1)
                y = df['Outcome']
            elif disease == "Heart_Disease":
                df = pd.read_csv(DATA_DIR / "heart_disease.csv")
                X = df.drop('target', axis=1)
                y = df['target']
            else:  # Parkinson's
                df = pd.read_csv(DATA_DIR / "parkinsons.csv")
                X = df.drop(['name', 'status'], axis=1)
                y = df['status']
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)
            
            # Display metrics
            st.subheader(f"Model Evaluation - {disease}")
            metrics = plot_metrics(y, y_pred, y_prob, disease)
            
            # Display feature importance if available
            if hasattr(model, 'feature_importances_') and feature_names is not None:
                st.subheader("Feature Importance")
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Create a figure for feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(importances)), importances[indices])
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels([feature_names[i] for i in indices], 
                                  rotation=45, ha="right")
                ax.set_title("Feature Importances")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")

# Add some styling
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e6f7ff;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)