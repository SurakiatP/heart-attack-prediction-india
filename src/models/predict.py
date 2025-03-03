import os
import sys
import pickle
import pandas as pd

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import modules from the src package
from src.data.load_data import load_data
from src.features.build_features import build_features
from src.models.train_model import encode_categorical_features  # Import the encoding function

def load_model(model_path):
    """
    Load a trained model from a pickle file.
    
    Parameters:
        model_path (str): Path to the pickle file.
    
    Returns:
        model: The loaded model.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, input_data):
    """
    Make predictions using the provided model.
    
    Parameters:
        model: The trained model.
        input_data (DataFrame): DataFrame with features matching the training data.
    
    Returns:
        array: Predictions from the model.
    """
    return model.predict(input_data)

if __name__ == "__main__":
    
    sample_data = {
        "Age": [45, 60],
        "Gender": ["Male", "Female"],
        "Diabetes": [1, 0],
        "Hypertension": [0, 1],
        "Obesity": [1, 1],
        "Smoking": [1, 0],
        "Alcohol_Consumption": [0, 1],
        "Physical_Activity": [2, 3],
        "Diet_Score": [6, 7],
        "Cholesterol_Level": [220, 280],
        "Triglyceride_Level": [180, 210],
        "LDL_Level": [100, 130],
        "HDL_Level": [50, 45],
        "Systolic_BP": [120, 140],
        "Diastolic_BP": [80, 90],
        "Air_Pollution_Exposure": [1, 1],
        "Family_History": [0, 1],
        "Stress_Level": [5, 8],
        "Healthcare_Access": [1, 0],
        "Heart_Attack_History": [0, 1],
        "Emergency_Response_Time": [15, 30],
        "Annual_Income": [50000, 300000],
        "Health_Insurance": [1, 0]
    }

    df_test = pd.DataFrame(sample_data)

    df_test = build_features(df_test)

    df_test = encode_categorical_features(df_test)
    
    # Define the model path
    model_path = os.path.join("models", "rf_model.pkl")
    
    # Load the trained model
    model = load_model(model_path)
    
    # Make predictions on the preprocessed input data
    predictions = predict(model, df_test)
    print("Predictions:", predictions)
