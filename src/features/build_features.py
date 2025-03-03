import os
import sys
import pandas as pd

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.load_data import load_data  # Import after modifying sys.path

def build_features(df):
    """
    Create new features from the DataFrame:
      - Remove identifier columns (e.g., Patient_ID).
      - Create an interaction feature: Age_Hypertension = Age * Hypertension.
    
    Parameters:
        df (DataFrame): Input data.
    
    Returns:
        DataFrame: DataFrame with new features.
    """

    df_feature = df.drop(columns=["Patient_ID", "State_Name", "Emergency_Response_Time", "Annual_Income"])
    df_features["Age_Hypertension"] = df_features["Age"] * df_feature["Hypertension"]

    return df_feature

if __name__ == "__main__":
    df = load_data()  # Load data using the correct function
    df_features = build_features(df)
    print(df_features.head())
