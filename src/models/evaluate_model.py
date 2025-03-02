import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.load_data import load_data
from src.features.build_features import build_features
# Optionally, import encoding function if needed for evaluation:
from src.models.train_model import encode_categorical_features

def evaluate_model(model_path):
    """
    Evaluate the trained model on the dataset.

    Parameters:
        model_path (str): Path to the trained model.
    
    Prints:
        Confusion matrix and classification report.
    """
    # Load and preprocess the data
    df = load_data()
    df = build_features(df)
    df = encode_categorical_features(df)  # Ensure categorical variables are encoded
    
    # Separate features and target variable
    X = df.drop('Heart_Attack_Risk', axis=1)
    y = df['Heart_Attack_Risk']

    # Load the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions and print evaluation metrics
    y_pred = model.predict(X)
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    model_path = os.path.join("models", "rf_model.pkl")
    evaluate_model(model_path)
