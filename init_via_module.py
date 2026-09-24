#!/usr/bin/env python3
"""Initialize data by running the prediction module"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import and run the module's main function with initialization
from src.prediction import F1ChampionshipPredictor

def main():
    print("Initializing F1 Championship Predictor data using module approach...")
    predictor = F1ChampionshipPredictor()
    # Just initialize the data - don't run full pipeline
    predictor.initialize_data(start_year=2015)
    print("Data initialization complete!")

if __name__ == "__main__":
    main()