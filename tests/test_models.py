import os
import unittest
import pandas as pd
from src.models.train_model import train_model, load_model
from src.models.predict import predict
from src.features.build_features import build_features  # import build_features for processing

class TestModels(unittest.TestCase):
    def test_train_and_predict(self):
        # Create a small dummy dataset
        df = pd.DataFrame({
            "Age": [40, 50, 60, 70],
            "Diabetes": [0, 1, 0, 1],
            "Hypertension": [1, 0, 1, 0],
            "Heart_Attack_Risk": [0, 1, 0, 1]
        })
        test_csv = "tests/dummy_data.csv"
        df.to_csv(test_csv, index=False)

        model_path = "tests/dummy_model.pkl"
        # Train the model using the dummy dataset
        model = train_model(test_csv, model_path)
        self.assertIsNotNone(model)

        # Load the trained model
        loaded_model = load_model(model_path)

        # Process the dummy dataset with build_features so that it includes 'Age_Hypertension'
        df_processed = build_features(df.copy())

        # Now drop the target variable before prediction
        preds = predict(loaded_model, df_processed.drop("Heart_Attack_Risk", axis=1))
        self.assertEqual(len(preds), df.shape[0])

        # Clean up temporary files
        os.remove(test_csv)
        os.remove(model_path)

if __name__ == '__main__':
    unittest.main()
