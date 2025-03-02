import unittest
import pandas as pd
from src.features.build_features import build_features
import os
import sys

# Append the project root to sys.path (one directory up from tests)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestBuildFeatures(unittest.TestCase):
    def test_build_features(self):
        # Create a sample DataFrame
        df = pd.DataFrame({
            "Patient_ID": [1, 2],
            "Age": [40, 50],
            "Hypertension": [1, 0],
            "Heart_Attack_Risk": [0, 1]
        })
        df_features = build_features(df)
        self.assertIn("Age_Hypertension", df_features.columns)

if __name__ == '__main__':
    unittest.main()
