#!/usr/bin/env python3
"""Script to run data initialization using the proper path setup"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prediction import F1ChampionshipPredictor

def main():
    print("Initializing F1 Championship Predictor data...")
    predictor = F1ChampionshipPredictor()
    predictor.initialize_data(start_year=2015)
    print("Data initialization complete!")

if __name__ == "__main__":
    main()