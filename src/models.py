"""
Models module for F1 Championship Predictor.
Handles model training, prediction, and evaluation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1Models:
    """Handles F1 prediction models."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        os.makedirs(model_dir, exist_ok=True)

    def prepare_features(self, df: pd.DataFrame, target_column: str = None) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """Prepare features for modeling."""
        if df.empty:
            return np.array([]), None, []

        # Exclude non-feature columns
        exclude_cols = [
            'year', 'round', 'driver_id', 'constructor_id', 'teammate_id',
            'actual_finish_position', 'actual_points', 'actual_dnf', 'actual_positions_gained',
            'championship_position', 'championship_points', 'championship_wins',
            'constructor_championship_position', 'constructor_championship_points', 'constructor_wins',
            'championship_leader_points', 'championship_leader_id', 'points_gap_to_leader',
            'circuit_name', 'location', 'country'
        ]

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Handle missing values - fill numeric columns with median, categorical with mode
        df_clean = df.copy()
        for col in feature_cols:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                # For categorical/object types, fill with mode or a default
                mode_val = df_clean[col].mode()
                df_clean[col] = df_clean[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'unknown')

        # Ensure we only use numeric features for now (can be extended for categorical)
        numeric_features = []
        for col in feature_cols:
            if df_clean[col].dtype in ['float64', 'int64']:
                numeric_features.append(col)
            else:
                # Try to convert to numeric
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    if not df_clean[col].isna().all():
                        numeric_features.append(col)
                except:
                    logger.warning(f"Dropping non-numeric column: {col}")

        feature_cols = numeric_features
        X = df_clean[feature_cols].values

        # Get target if specified
        y = None
        if target_column and target_column in df_clean.columns:
            y = df_clean[target_column].values
            # Handle missing targets
            if y is not None:
                # For classification, we might need to handle this differently
                pass

        self.feature_columns = feature_cols
        return X, y, feature_cols

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> LogisticRegression:
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression...")
        model = LogisticRegression(random_state=42, max_iter=1000, **kwargs)
        model.fit(X_train, y_train)
        return model

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestClassifier:
        """Train Random Forest model."""
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
        model.fit(X_train, y_train)
        return model

    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> GradientBoostingClassifier:
        """Train Gradient Boosting model."""
        logger.info("Training Gradient Boosting...")
        model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
            **kwargs
        )
        model.fit(X_train, y_train)
        return model

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        logger.info("Training XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            **kwargs
        )
        model.fit(X_train, y_train)
        return model

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> lgb.LGBMClassifier:
        """Train LightGBMA model."""
        logger.info("Training LightGBM...")
        model = lgb.LGBMClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **kwargs
        )
        model.fit(X_train, y_train)
        return model

    def train_catboost(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> cb.CatBoostClassifier:
        """Train CatBoost model."""
        logger.info("Training CatBoost...")
        model = cb.CatBoostClassifier(
            iterations=100,
            random_seed=42,
            verbose=False,
            **kwargs
        )
        model.fit(X_train, y_train)
        return model

    def train_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Any:
        """Train a model of the specified type."""
        if model_type == 'logistic_regression':
            model = self.train_logistic_regression(X_train, y_train, **kwargs)
        elif model_type == 'random_forest':
            model = self.train_random_forest(X_train, y_train, **kwargs)
        elif model_type == 'gradient_boosting':
            model = self.train_gradient_boosting(X_train, y_train, **kwargs)
        elif model_type == 'xgboost':
            model = self.train_xgboost(X_train, y_train, **kwargs)
        elif model_type == 'lightgbm':
            model = self.train_lightgbm(X_train, y_train, **kwargs)
        elif model_type == 'catboost':
            model = self.train_catboost(X_train, y_train, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.models[model_type] = model
        return model

    def predict_proba(self, model_type: str, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from a trained model."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")

        model = self.models[model_type]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]  # Return probability of positive class
        else:
            # For models without predict_proba, use decision function or predict
            try:
                return model.predict_proba(X)[:, 1]
            except:
                # Fallback to predict
                preds = model.predict(X)
                # Convert to probabilities (crude approximation)
                return np.where(preds == 1, 0.8, 0.2)

    def predict(self, model_type: str, X: np.ndarray) -> np.ndarray:
        """Get predictions from a trained model."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")

        return self.models[model_type].predict(X)

    def evaluate_model(self, model_type: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a trained model."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")

        model = self.models[model_type]
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

        # Try to get ROC AUC if probabilities available
        try:
            y_pred_proba = self.predict_proba(model_type, X_test)
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.0

        return metrics

    def evaluate_regression(self, model_type: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a regression model."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")

        model = self.models[model_type]
        y_pred = model.predict(X_test)

        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }

        return metrics

    def save_model(self, model_type: str, filepath: str = None) -> None:
        """Save a trained model to disk."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")

        if filepath is None:
            filepath = f"{self.model_dir}/{model_type}_model.pkl"

        joblib.dump(self.models[model_type], filepath)
        logger.info(f"Saved {model_type} model to {filepath}")

    def load_model(self, model_type: str, filepath: str = None) -> Any:
        """Load a trained model from disk."""
        if filepath is None:
            filepath = f"{self.model_dir}/{model_type}_model.pkl"

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model = joblib.load(filepath)
        self.models[model_type] = model
        logger.info(f"Loaded {model_type} model from {filepath}")
        return model

    def save_all_models(self) -> None:
        """Save all trained models."""
        for model_type in self.models.keys():
            self.save_model(model_type)

    def load_all_models(self) -> None:
        """Load all saved models."""
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('_model.pkl')]
        for model_file in model_files:
            model_type = model_file.replace('_model.pkl', '')
            try:
                self.load_model(model_type)
            except Exception as e:
                logger.warning(f"Could not load model {model_type}: {e}")

def create_prediction_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create various prediction targets from race results."""
    if df.empty:
        return df

    df_with_targets = df.copy()

    # Binary classification targets
    df_with_targets['won_race'] = (df_with_targets['actual_finish_position'] == 1).astype(int)
    df_with_targets['podium_finish'] = (df_with_targets['actual_finish_position'] <= 3).astype(int)
    df_with_targets['top10_finish'] = (df_with_targets['actual_finish_position'] <= 10).astype(int)
    df_with_targets['points_finish'] = (df_with_targets['actual_points'] > 0).astype(int)

    # Multi-class targets (position bins)
    df_with_targets['finish_position_bin'] = pd.cut(
        df_with_targets['actual_finish_position'],
        bins=[0, 3, 10, 20, 100],
        labels=[0, 1, 2, 3],  # Podium, Points, Outside points, DNF/poor
        include_lowest=True
    )

    # Regression targets
    df_with_targets['target_points'] = df_with_targets['actual_points']
    df_with_targets['target_finish_position'] = df_with_targets['actual_finish_position']

    return df_with_targets

def main():
    """Main function for testing models."""
    print("F1 Models module ready")

if __name__ == "__main__":
    main()