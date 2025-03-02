# import os
# import sys
# import pickle
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# from src.data.load_data import load_data
# from src.features.build_features import build_features

# def encode_categorical_features(df):
#     """
#     Encode categorical features using one-hot encoding.
#     """
#     categorical_columns = ["Gender", "State_Name"]
    
#     existing_cats = [col for col in categorical_columns if col in df.columns]
#     if existing_cats:
#         df = pd.get_dummies(df, columns=existing_cats, drop_first=True)
#     return df

# def train_model(data_path, model_path):
#     """
#     Train a RandomForestClassifier model on the dataset.
    
#     Parameters:
#         data_path (str): Path to the CSV file for training.
#         model_path (str): Path to save the trained model.
    
#     Returns:
#         model: The trained RandomForest model.
#     """

#     df = load_data(data_path)
    
#     df = build_features(df)
    
#     # Encode categorical features
#     df = encode_categorical_features(df)
    
#     if "Heart_Attack_Risk" not in df.columns:
#         raise ValueError("Missing target variable 'Heart_Attack_Risk' in dataset!")
    
#     X = df.drop("Heart_Attack_Risk", axis=1)
#     y = df["Heart_Attack_Risk"]
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
    
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print("Test Accuracy: {:.2f}%".format(acc * 100))
    
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     with open(model_path, 'wb') as f:
#         pickle.dump(model, f)
    
#     return model

# def load_model(model_path):
#     """
#     Load a trained model from a pickle file.
    
#     Parameters:
#         model_path (str): Path to the pickle file.
    
#     Returns:
#         model: The loaded model.
#     """
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)
#     return model

# if __name__ == "__main__":
#     model_path = os.path.join("models", "rf_model.pkl")
#     train_model("heart_attack_prediction_india.csv", model_path)
import os
import sys
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the MLflow tracking URI to a valid http URL (make sure MLflow server is running)
mlflow.set_tracking_uri("http://localhost:5000")

# เพิ่ม project root เข้าไปใน sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.load_data import load_data
from src.features.build_features import build_features

def encode_categorical_features(df):
    """
    Encode categorical features using one-hot encoding.
    """
    categorical_columns = ["Gender", "State_Name"]
    existing_cats = [col for col in categorical_columns if col in df.columns]
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, drop_first=True)
    return df

def train_model(data_path, model_path):
    """
    Train a RandomForestClassifier model on the dataset, and log metrics/models to MLflow.
    
    Parameters:
        data_path (str): Path to the CSV file for training.
        model_path (str): Path to save the trained model (.pkl).
    
    Returns:
        model: The trained RandomForest model.
    """

    # เริ่มต้น MLflow run
    with mlflow.start_run(run_name="heart_attack_train"):
        # 1) โหลดข้อมูล
        df = load_data(data_path)
        
        # 2) สร้าง feature ใหม่
        df = build_features(df)
        
        # 3) เข้ารหัส categorical features
        df = encode_categorical_features(df)
        
        # ตรวจสอบว่ามีคอลัมน์เป้าหมายหรือไม่
        if "Heart_Attack_Risk" not in df.columns:
            raise ValueError("Missing target variable 'Heart_Attack_Risk' in dataset!")
        
        # แยก features และ target
        X = df.drop("Heart_Attack_Risk", axis=1)
        y = df["Heart_Attack_Risk"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4) สร้างและเทรนโมเดล
        n_estimators = 100
        random_state = 42
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        
        # 5) ประเมินผล
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Test Accuracy: {:.2f}%".format(acc * 100))
        
        # ---- (A) Log Parameters & Metrics ลงใน MLflow ----
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", acc)
        
        # ---- (B) บันทึกโมเดลลงไฟล์ .pkl (เพื่อให้ app.py ใช้งาน) ----
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # ---- (C) Log ตัวโมเดลลง MLflow ----
        mlflow.sklearn.log_model(model, artifact_path="model")
    
    return model

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

if __name__ == "__main__":
    model_path = os.path.join("models", "rf_model.pkl")
    train_model("heart_attack_prediction_india.csv", model_path)


