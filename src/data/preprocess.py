# import os
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from load_data import load_data  # Ensure this import is correct

# # Load the dataset (no need to pass file_path, it defaults correctly)
# df = load_data()  
# print("Data loaded successfully!")

# def preprocess_data(df):
#     """
#     Clean and preprocess the data:
#       - Fill missing values using forward fill.
#       - Encode categorical variables and scale numerical features.
    
#     Parameters:
#         df (DataFrame): Raw input data.
    
#     Returns:
#         tuple: Processed numpy array and the fitted preprocessor.
#     """
#     # ✅ Fix: Use .ffill() instead of fillna(method='ffill')
#     df = df.ffill()

#     # Define categorical features
#     categorical_features = ['Gender', 'State_Name']
    
#     # Assume all other columns (excluding Patient_ID and Heart_Attack_Risk) are numeric
#     numeric_features = [col for col in df.columns 
#                         if col not in categorical_features + ['Patient_ID', 'Heart_Attack_Risk']]
    
#     numeric_transformer = StandardScaler()

#     # ✅ Fix: Use sparse_output=False instead of sparse=False
#     categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numeric_features),
#             ('cat', categorical_transformer, categorical_features)
#         ]
#     )
    
#     processed_array = preprocessor.fit_transform(df)
#     return processed_array, preprocessor

# if __name__ == "__main__":
#     processed_data, preprocessor = preprocess_data(df)  # ✅ Use the `df` already loaded
#     print("Processed data shape:", processed_data.shape)

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from load_data import load_data  # Ensure this import is correct

# Load the dataset
df = load_data()  # Do NOT pass a path; load_data() จะจัดการ path เอง
print("Data loaded successfully!")

def preprocess_data(df):
    """
    Clean and preprocess the data:
      - Fill missing values using forward fill.
      - Encode categorical variables and scale numerical features.
    
    Parameters:
        df (DataFrame): Raw input data.
    
    Returns:
        tuple: Processed data (as numpy array) and the fitted preprocessor.
    """
    # Use ffill() instead of fillna(method='ffill') to avoid future warning
    df = df.ffill()
    
    # Define categorical features
    categorical_features = ['Gender', 'State_Name']
    # Assume all other columns (excluding Patient_ID and Heart_Attack_Risk) are numeric
    numeric_features = [col for col in df.columns 
                        if col not in categorical_features + ['Patient_ID', 'Heart_Attack_Risk']]
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    processed_array = preprocessor.fit_transform(df)
    return processed_array, preprocessor

if __name__ == "__main__":
    import os
    file_path = os.path.join("data", "raw", "heart_attack_prediction_india.csv")
    df = load_data(file_path)
    processed_data, preprocessor = preprocess_data(df)
    print("Processed data shape:", processed_data.shape)
    
    # Save processed data to CSV file (DVC expects output file 'data/processed/heart_attack_data.csv')
    output_path = os.path.join("data", "processed", "heart_attack_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Optionally, get feature names from preprocessor if available
    try:
        feature_names = preprocessor.get_feature_names_out()
        processed_df = pd.DataFrame(processed_data, columns=feature_names)
    except Exception:
        processed_df = pd.DataFrame(processed_data)
    
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
