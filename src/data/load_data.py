import os
import pandas as pd

def load_data(filename="heart_attack_prediction_india.csv"):
    """
    Load data from a CSV file.

    Parameters:
        filename (str): Path or filename of the CSV file.
    
    Returns:
        DataFrame: Loaded data as a pandas DataFrame.
    """

    if os.path.exists(filename):
        print(f"Loading from provided file path: {os.path.abspath(filename)}")
        return pd.read_csv(filename)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_path = os.path.join(project_root, "data", "raw", filename)
    print(f"Looking for file at: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: The file '{data_path}' was not found! Please check the file location.")
    return pd.read_csv(data_path)

if __name__ == "__main__":
    df = load_data()
    print(df.head())
