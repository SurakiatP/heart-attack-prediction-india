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
    # Load the data
    df = load_data()
    # Apply feature engineering
    df = build_features(df)
    # Encode categorical features to match training time encoding
    df = encode_categorical_features(df)
    
    # Separate features and target variable
    X = df.drop("Heart_Attack_Risk", axis=1)
    
    # Define the model path
    model_path = os.path.join("models", "rf_model.pkl")
    
    # Load the trained model
    model = load_model(model_path)
    
    # Make predictions on the preprocessed input data
    predictions = predict(model, X)
    print("Predictions:", predictions)
