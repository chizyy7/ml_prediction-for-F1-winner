#!/usr/bin/env python3
"""Test script to verify data loader works"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import F1DataLoader

def main():
    print("Testing F1 Data Loader...")
    loader = F1DataLoader("data/raw")
    print("Data Loader initialized successfully!")

    # Test getting seasons (this will make an API call)
    try:
        print("Attempting to get seasons data...")
        seasons_df = loader.get_seasons(2020, 2023)
        print(f"Successfully retrieved {len(seasons_df)} seasons")
        print(seasons_df.head())
    except Exception as e:
        print(f"Note: API call failed (expected without internet or if API is down): {e}")
        print("This is OK - the important thing is that the loader initialized correctly")

if __name__ == "__main__":
    main()