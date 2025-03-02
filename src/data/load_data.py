# import os
# import pandas as pd

# def load_data(filename="heart_attack_prediction_india.csv"):
#     """
#     Load data from a CSV file located in the data/raw directory.

#     Parameters:
#         filename (str): Name of the CSV file.

#     Returns:
#         DataFrame: Loaded data as a pandas DataFrame.
#     """
#     # Get the absolute path of the project root (two levels up from src/data/)
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

#     # Construct the correct absolute path to the data file
#     data_path = os.path.join(project_root, "data", "raw", filename)

#     # Print the expected path for debugging
#     print(f"Looking for file at: {data_path}")

#     # Check if the file exists before loading
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"Error: The file '{data_path}' was not found! Please check the file location.")

#     # Load and return the data
#     return pd.read_csv(data_path)

# if __name__ == "__main__":
#     df = load_data()
#     print(df.head())  # Display first 5 rows

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
    # ถ้าไฟล์ที่ระบุมีอยู่ใน path ที่ให้มา ให้โหลดตรงนั้น
    if os.path.exists(filename):
        print(f"Loading from provided file path: {os.path.abspath(filename)}")
        return pd.read_csv(filename)
    
    # ถ้าไม่พบ ให้สมมุติว่าเป็น filename ที่อยู่ใน data/raw จาก root ของโปรเจค
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_path = os.path.join(project_root, "data", "raw", filename)
    print(f"Looking for file at: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: The file '{data_path}' was not found! Please check the file location.")
    return pd.read_csv(data_path)

if __name__ == "__main__":
    df = load_data()
    print(df.head())
