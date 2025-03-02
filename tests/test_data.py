import unittest
import os
import pandas as pd
from src.data.load_data import load_data
import sys

# Append the project root to sys.path (one directory up from tests)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestLoadData(unittest.TestCase):
    def test_load_data(self):
        # Create a sample CSV file for testing
        sample_csv = "tests/sample_data.csv"
        df_sample = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df_sample.to_csv(sample_csv, index=False)
        df_loaded = load_data(sample_csv)
        self.assertEqual(df_loaded.shape, df_sample.shape)
        os.remove(sample_csv)

if __name__ == '__main__':
    unittest.main()
