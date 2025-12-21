import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import clean_data

class TestPreprocessing(unittest.TestCase):
    def test_clean_data_removes_duplicates(self):
        df = pd.DataFrame({
            'a': [1, 1, 2],
            'b': [3, 3, 4]
        })
        cleaned = clean_data(df)
        self.assertEqual(len(cleaned), 2)

    def test_clean_data_removes_nans(self):
        df = pd.DataFrame({
            'a': [1, np.nan, 2],
            'b': [3, 4, 5]
        })
        cleaned = clean_data(df)
        self.assertEqual(len(cleaned), 2)

if __name__ == '__main__':
    unittest.main()
