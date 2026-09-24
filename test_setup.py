"""
Test script to verify the F1 Championship Predictor setup
"""

import sys
import os
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        # Test data loader
        from src.data_loader import F1DataLoader
        print("[OK] data_loader imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import data_loader: {e}")
        traceback.print_exc()
        return False

    try:
        # Test data cleaning
        from src.data_cleaning import F1DataCleaner
        print("[OK] data_cleaning imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import data_cleaning: {e}")
        traceback.print_exc()
        return False

    try:
        # Test feature engineering
        from src.feature_engineering import F1FeatureEngineer
        print("[OK] feature_engineering imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import feature_engineering: {e}")
        traceback.print_exc()
        return False

    try:
        # Test models
        from src.models import F1Models
        print("[OK] models imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import models: {e}")
        traceback.print_exc()
        return False

    try:
        # Test evaluation
        from src.evaluation import F1Evaluator
        print("[OK] evaluation imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import evaluation: {e}")
        traceback.print_exc()
        return False

    try:
        # Test simulation
        from src.simulation import F1Simulation
        print("[OK] simulation imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import simulation: {e}")
        traceback.print_exc()
        return False

    try:
        # Test explainability
        from src.explainability import F1Explainability
        print("[OK] explainability imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import explainability: {e}")
        traceback.print_exc()
        return False

    try:
        # Test prediction pipeline
        from src.prediction import F1ChampionshipPredictor
        print("[OK] prediction imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import prediction: {e}")
        traceback.print_exc()
        return False

    try:
        # Test Streamlit app (this might fail if streamlit not installed)
        sys.path.append('app')
        import streamlit_app
        print("[OK] streamlit_app imported successfully")
    except Exception as e:
        print(f"[WARN] Streamlit app import issue (may be okay if streamlit not installed): {e}")
        # Don't return False for this as it's expected in basic env

    print("All core imports successful!")
    return True

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")

    try:
        # Test data loader initialization
        from src.data_loader import F1DataLoader
        loader = F1DataLoader("data/raw")
        print("[OK] F1DataLoader initialized")
    except Exception as e:
        print(f"[FAIL] F1DataLoader initialization failed: {e}")
        return False

    try:
        # Test feature engineer initialization
        from src.feature_engineering import F1FeatureEngineer
        engineer = F1FeatureEngineer()
        print("[OK] F1FeatureEngineer initialized")
    except Exception as e:
        print(f"[FAIL] F1FeatureEngineer initialization failed: {e}")
        return False

    try:
        # Test models initialization
        from src.models import F1Models
        models = F1Models("models")
        print("[OK] F1Models initialized")
    except Exception as e:
        print(f"[FAIL] F1Models initialization failed: {e}")
        return False

    print("Basic functionality tests passed!")
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("F1 Championship Predictor Setup Test")
    print("=" * 50)

    success = True

    success &= test_imports()
    success &= test_basic_functionality()

    print("\n" + "=" * 50)
    if success:
        print("ALL TESTS PASSED! Setup is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Initialize data: python src/prediction.py --init")
        print("3. Run full pipeline: python src/prediction.py")
        print("4. Launch dashboard: streamlit run app/streamlit_app.py")
    else:
        print("SOME TESTS FAILED! Please check the output above.")
    print("=" * 50)

    return success

if __name__ == "__main__":
    main()