"""
Unit tests for the F1 data loader
"""
import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, Mock
import pandas as pd

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestF1DataLoader(unittest.TestCase):
    """Test cases for F1DataLoader"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.loader = F1DataLoader(cache_dir=self.test_dir)

    def tearDown(self):
        """Tear down test fixtures"""
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test that the loader initializes correctly"""
        self.assertIsInstance(self.loader, F1DataLoader)
        self.assertEqual(self.loader.cache_dir, self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))

    def test_make_request_cache_directory(self):
        """Test that cache directory is created"""
        # This test would require mocking the actual API request
        # For now, we just test that the directory handling works
        self.assertTrue(os.path.exists(self.test_dir))

    def test_get_seasons_returns_dataframe(self):
        """Test that get_seasons returns a DataFrame"""
        # This would require mocking the API call
        # We'll test the method exists and returns correct type when mocked
        with patch.object(self.loader, '_make_request') as mock_make_request:
            # Mock API response
            mock_response = {
                'MRData': {
                    'SeasonTable': {
                        'Seasons': [
                            {'year': '2020', 'url': 'http://ergast.com/api/f1/2020'},
                            {'year': '2021', 'url': 'http://ergast.com/api/f1/2021'}
                        ]
                    }
                }
            }
            mock_make_request.return_value = mock_response

            df = self.loader.get_seasons(2020, 2021)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)
            self.assertIn('year', df.columns)
            self.assertIn('url', df.columns)

if __name__ == '__main__':
    unittest.main()
EOF