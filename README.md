# **Heart_Attack_Prediction_India**

A **Machine Learning Pipeline** to predict heart attack risk based on patient data using **DVC**, **MLflow**, and **GitHub Actions**.

---

## **📂 Project Structure**
```
Heart_Attack_Prediction/
│── data/                     # Data directory (managed by DVC)
│   ├── raw/                  # Raw dataset (tracked with DVC)
│   ├── processed/            # Processed dataset
│── models/                   # Trained model storage
│── notebooks/                # Jupyter notebooks for EDA
│── src/                      # Source code
│   ├── data/                 # Data processing modules
│   │   ├── load_data.py      # Loads dataset from CSV
│   │   ├── preprocess.py     # Preprocesses data (scaling, encoding)
│   ├── features/             # Feature engineering
│   │   ├── build_features.py # Creates new features
│   ├── models/               # Model training & evaluation
│   │   ├── train_model.py    # Trains a RandomForest model
│   │   ├── evaluate_model.py # Evaluates trained model
│   │   ├── predict.py        # Uses trained model for predictions
│   ├── utils/                # Helper functions
│   │   ├── helpers.py        # Utility functions
│   ├── visualization/        # Visualization module
│   │   ├── visualize.py      # Generates plots for data & model results
│── .dvc/                     # DVC tracking files
│── .github/workflows/        # GitHub Actions CI/CD pipelines
│── .gitignore                # Ignore unnecessary files
│── dvc.yaml                  # DVC pipeline configuration
│── requirements.txt          # Python dependencies
│── README.md                 # Project documentation
```

---

---

## **📌 Feature Descriptions**
This project uses patient data with the following features:

| Feature | Description |
|---------|------------|
| `Age` | Age of the patient |
| `Gender` | Gender (Male/Female) |
| `Diabetes` | Whether the patient has diabetes (0 = No, 1 = Yes) |
| `Hypertension` | Whether the patient has hypertension (0 = No, 1 = Yes) |
| `Obesity` | Obesity status (0 = No, 1 = Yes) |
| `Smoking` | Smoking habits (0 = No, 1 = Yes) |
| `Alcohol_Consumption` | Alcohol consumption (0 = No, 1 = Yes) |
| `Physical_Activity` | Level of physical activity |
| `Diet_Score` | Score representing diet quality |
| `Cholesterol_Level` | Measured cholesterol level |
| `LDL_Level` | Low-density lipoprotein (LDL) cholesterol level |
| `HDL_Level` | High-density lipoprotein (HDL) cholesterol level |
| `Systolic_BP` | Systolic blood pressure level |
| `Diastolic_BP` | Diastolic blood pressure level |
| `Air_Pollution_Exposure` | Level of air pollution exposure |
| `Family_History` | Family history of heart disease (0 = No, 1 = Yes) |
| `Stress_Level` | Stress level score |
| `Healthcare_Access` | Access to healthcare services (0 = No, 1 = Yes) |
| `Heart_Attack_History` | Previous history of heart attack (0 = No, 1 = Yes) |
| `Emergency_Response_Time` | Time taken for emergency response |
| `Annual_Income` | Annual income of the patient |
| `Health_Insurance` | Whether the patient has health insurance (0 = No, 1 = Yes) |
| `Heart_Attack_Risk` | Target variable: Predicted risk of heart attack (0 = Low, 1 = High) |

---

## **🧠 Model Used**
This project utilizes a **Random Forest Classifier** as the primary model for heart attack risk prediction. 

### **Why Random Forest?**
- Handles both categorical and numerical data efficiently
- Reduces overfitting compared to a single decision tree
- Provides feature importance insights
- Works well with imbalanced datasets

Hyperparameters used:
- `n_estimators`: 100
- `random_state`: 42

The trained model is logged and tracked using **MLflow**.

---

---

## **⚙️ Installation**
Clone this repository and set up a virtual environment:
```bash
git clone https://github.com/SurakiatP/Heart_Attack_Prediction.git
cd Heart_Attack_Prediction
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

---

## **🚀 How to Run the Project**
### **1️⃣ Setup DVC (Data Version Control)**
```bash
dvc pull  # Fetch data files from remote storage
```

### **2️⃣ Reproduce the pipeline (Data Preprocessing → Training → Evaluation → Visualization)**
```bash
dvc repro
```

### **3️⃣ Start MLflow UI (for experiment tracking)**
```bash
mlflow ui
```
Then open `http://localhost:5000` in your browser.

### **4️⃣ Make Predictions**
```bash
python src/models/predict.py
```

### **5️⃣ Deploy using Flask**
```bash
python app.py
```
Then visit `http://127.0.0.1:8000/predict` and send a POST request with JSON input.

---

## **📌 Workflow & File Responsibilities**
### **1️⃣ Data Processing**
| File | Description |
|------|------------|
| `src/data/load_data.py` | Loads raw dataset from CSV |
| `src/data/preprocess.py` | Handles missing values, encodes categorical features, and scales numerical features |

### **2️⃣ Feature Engineering**
| File | Description |
|------|------------|
| `src/features/build_features.py` | Creates new features (e.g., `Age_Hypertension`) |

### **3️⃣ Model Training & Evaluation**
| File | Description |
|------|------------|
| `src/models/train_model.py` | Trains a **RandomForest** model and logs to **MLflow** |
| `src/models/evaluate_model.py` | Evaluates the trained model and reports metrics |
| `src/models/predict.py` | Loads the trained model and makes predictions |

### **4️⃣ Visualization**
| File | Description |
|------|------------|
| `src/visualization/visualize.py` | Generates plots such as feature importance |

### **5️⃣ Deployment & API**
| File | Description |
|------|------------|
| `app.py` | Flask API for serving predictions |

---

**Workflow Recap**:
- **`preprocess`** → Creates a processed dataset
- **`train`** → Trains the RandomForest model
- **`visualize`** → Produces key visualization files
- **`evaluate`** (optional) → Evaluates the trained model
- **`predict`** (optional) → Generates predictions on new data
- **`app.py`** (optional) → Deploys a Flask API for real-time prediction

---

## **⚡ DVC Pipeline (`dvc.yaml`)**
The project is structured into **stages** using DVC:
```yaml
stages:
  preprocess:
    cmd: python src/data/preprocess.py
    deps:
      - data/raw/heart_attack_prediction_india.csv
      - src/data/preprocess.py
    outs:
      - data/processed/heart_attack_data.csv

  train:
    cmd: python src/models/train_model.py
    deps:
      - data/processed/heart_attack_data.csv
      - src/models/train_model.py
    outs:
      - models/rf_model.pkl

  visualize:
    cmd: python src/visualization/visualize.py
    deps:
      - models/rf_model.pkl
      - src/visualization/visualize.py
    outs:
      - plots/feature_importance.png
```

---

## **🛠 GitHub Actions (CI/CD)**
This project includes a **CI/CD pipeline** in `.github/workflows/ci.yml` to:
- Install dependencies
- Run **DVC pipeline**
- Execute **unit tests** using `pytest`
- Log experiment results

### **Run tests locally**
```bash
pytest tests/
```

---

## **📌 Key Features**
✅ **Automated ML pipeline** using **DVC**  
✅ **Experiment Tracking** with **MLflow**  
✅ **Model Deployment** using **Flask API**  
✅ **CI/CD integration** with **GitHub Actions**  

---

## **📜 License**
This project is licensed under the MIT License.

---

## **👨‍💻 Contributors**
- **Surakiat Kansa-ard** - _ML Engineer_

If you find this project useful, feel free to ⭐️ star the repository and contribute! 🚀

---

## **📞 Contact**
For questions, please contact:  
📧 Email: surakiat.0723@gmail.com  
🔗 GitHub: [SurakiatP](https://github.com/SurakiatP)

---

### **📢 Final Notes**
This README provides an **overview** of the **project structure, pipeline stages, and how to use the system**.  
It also includes **step-by-step instructions** for **running the pipeline, tracking experiments, and deploying the model**.  

🔥 **Now you can use this as a solid documentation for your project!** 🔥


