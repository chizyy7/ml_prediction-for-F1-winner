"""
Main prediction pipeline for F1 Championship Predictor.
Orchestrates the entire ML workflow from data to predictions.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from .data_loader import F1DataLoader
from .data_cleaning import F1DataCleaner, clean_all_data
from .feature_engineering import F1FeatureEngineer
from .models import F1Models, create_prediction_targets
from .evaluation import F1Evaluator
from .simulation import F1Simulation
from .explainability import F1Explainability

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1ChampionshipPredictor:
    """Main prediction pipeline for F1 championship outcomes."""

    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.raw_data_dir = os.path.join(data_dir, "raw")
        self.processed_data_dir = os.path.join(data_dir, "processed")

        # Initialize components
        self.data_loader = F1DataLoader(self.raw_data_dir)
        self.data_cleaner = F1DataCleaner()
        self.feature_engineer = F1FeatureEngineer()
        self.models = F1Models(model_dir)
        self.evaluator = F1Evaluator()
        self.simulator = F1Simulation()
        self.explainability = F1Explainability()

        # Ensure directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Storage for processed data
        self.data = {}
        self.features = None
        self.trained_models = {}
        self.predictions = None
        self.championship_simulation = None

    def initialize_data(self, start_year: int = 2015, end_year: Optional[int] = None) -> None:
        """Initialize the data pipeline by downloading historical data."""
        logger.info("Initializing data pipeline...")
        self.data_loader.download_historical_data(start_year=start_year, end_year=end_year)
        logger.info("Data initialization complete")

    def clean_data(self) -> None:
        """Clean all raw data."""
        logger.info("Cleaning data...")
        clean_all_data(self.raw_data_dir, self.processed_data_dir)
        logger.info("Data cleaning complete")

    def load_processed_data(self) -> Dict[str, pd.DataFrame]:
        """Load all processed data into memory."""
        logger.info("Loading processed data...")
        self.data = {}

        # Define expected data files
        data_files = {
            'races': 'clean_races_*.parquet',
            'results': 'clean_results_*.parquet',
            'qualifying': 'clean_qualifying_*.parquet',
            'driver_standings': 'clean_driver_standings_*.parquet',
            'constructor_standings': 'clean_constructor_standings_*.parquet',
            'circuits': 'clean_circuits.parquet'
        }

        for data_key, pattern in data_files.items():
            import glob
            files = glob.glob(os.path.join(self.processed_data_dir, pattern))
            if files:
                # If multiple files (like per-year), combine them
                if len(files) > 1 and '*' in pattern:
                    dfs = []
                    for file in files:
                        try:
                            df = pd.read_parquet(file)
                            dfs.append(df)
                        except Exception as e:
                            logger.warning(f"Could not load {file}: {e}")
                    if dfs:
                        self.data[data_key] = pd.concat(dfs, ignore_index=True)
                        logger.info(f"Loaded {len(self.data[data_key])} records for {data_key}")
                else:
                    # Single file
                    try:
                        self.data[data_key] = pd.read_parquet(files[0])
                        logger.info(f"Loaded {len(self.data[data_key])} records for {data_key}")
                    except Exception as e:
                        logger.error(f"Error loading {files[0]}: {e}")
                        self.data[data_key] = pd.DataFrame()
            else:
                logger.warning(f"No files found for pattern: {pattern}")
                self.data[data_key] = pd.DataFrame()

        return self.data

    def engineer_features(self) -> pd.DataFrame:
        """Engineer features from loaded data."""
        logger.info("Engineering features...")
        if not self.data:
            self.load_processed_data()

        self.features = self.feature_engineer.engineer_all_features(self.data)
        logger.info(f"Feature engineering complete: {len(self.features)} samples with {len(self.features.columns)} features")
        return self.features

    def prepare_training_data(self, target_type: str = 'win_race',
                            test_size: float = 0.2,
                            chronological_split: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare data for model training."""
        logger.info("Preparing training data...")

        if self.features is None or self.features.empty:
            self.engineer_features()

        df = self.features.copy()

        # Create target variable
        df_with_targets = create_prediction_targets(df)

        # Select target column
        target_mapping = {
            'win_race': 'won_race',
            'podium': 'podium_finish',
            'top10': 'top10_finish',
            'points': 'points_finish'
        }

        target_col = target_mapping.get(target_type, 'won_race')

        if target_col not in df_with_targets.columns:
            logger.warning(f"Target column {target_col} not found, creating it...")
            df_with_targets = create_prediction_targets(df)
            target_col = target_mapping.get(target_type, 'won_race')

        # Prepare features
        X, y, feature_names = self.models.prepare_features(df_with_targets, target_col)

        if len(X) == 0:
            logger.error("No features prepared for training")
            return np.array([]), np.array([]), np.array([]), np.array([]), []

        # Split data
        if chronological_split and 'year' in df_with_targets.columns and 'round' in df_with_targets.columns:
            # Chronological split: train on earlier years, test on later years
            df_with_targets = df_with_targets.sort_values(['year', 'round']).reset_index(drop=True)
            split_idx = int(len(df_with_targets) * (1 - test_size))

            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx] if y is not None else None
            y_test = y[split_idx:] if y is not None else None

            logger.info(f"Chronological split: {len(X_train)} train, {len(X_test)} test samples")
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            else:
                X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
                y_train, y_test = None, None

            logger.info(f"Random split: {len(X_train)} train, {len(X_test)} test samples")

        return X_train, X_test, y_train, y_test, feature_names

    def train_models(self, target_type: str = 'win_race',
                    model_types: List[str] = None) -> Dict[str, Any]:
        """Train multiple models for comparison."""
        logger.info("Training models...")

        if model_types is None:
            model_types = ['logistic_regression', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost']

        # Prepare training data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_training_data(target_type)

        if len(X_train) == 0 or y_train is None:
            logger.error("Failed to prepare training data")
            return {}

        # Scale features for linear models
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Store scalers
        self.models.scalers['standard'] = scaler

        # Train each model
        trained_models = {}

        for model_type in model_types:
            try:
                logger.info(f"Training {model_type}...")

                # Use scaled data for linear models, original for tree-based
                if model_type == 'logistic_regression':
                    X_train_use = X_train_scaled
                    X_test_use = X_test_scaled
                else:
                    X_train_use = X_train
                    X_test_use = X_test

                # Train model
                model = self.models.train_model(model_type, X_train_use, y_train)

                # Evaluate model
                if y_test is not None:
                    metrics = self.models.evaluate_model(model_type, X_test_use, y_test)
                    logger.info(f"{model_type} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")

                # Save model
                self.models.save_model(model_type)

                trained_models[model_type] = {
                    'model': model,
                    'feature_names': feature_names,
                    'X_train': X_train_use,
                    'X_test': X_test_use,
                    'y_train': y_train,
                    'y_test': y_test,
                    'metrics': self.models.evaluate_model(model_type, X_test_use, y_test) if y_test is not None else {}
                }

            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                continue

        self.trained_models = trained_models
        logger.info(f"Model training complete: {len(trained_models)} models trained")
        return trained_models

    def evaluate_models(self) -> Dict[str, Any]:
        """Evaluate all trained models."""
        logger.info("Evaluating models...")

        if not self.trained_models:
            logger.warning("No trained models to evaluate")
            return {}

        evaluation_results = {}

        for model_type, model_info in self.trained_models.items():
            try:
                if model_info['y_test'] is not None:
                    # Classification metrics
                    y_pred = model_info['model'].predict(model_info['X_test'])
                    y_pred_proba = None
                    if hasattr(model_info['model'], 'predict_proba'):
                        y_pred_proba = model_info['model'].predict_proba(model_info['X_test'])[:, 1]

                    metrics = self.evaluator.evaluate_classification(
                        model_info['y_test'], y_pred, y_pred_proba, model_type
                    )

                    # Add calibration evaluation
                    if y_pred_proba is not None:
                        calibration = self.evaluator.evaluate_calibration(
                            model_info['y_test'], y_pred_proba
                        )
                        metrics['calibration'] = calibration

                    evaluation_results[model_type] = metrics

            except Exception as e:
                logger.error(f"Error evaluating {model_type}: {e}")
                continue

        self.evaluator.evaluation_results = evaluation_results
        logger.info(f"Model evaluation complete: {len(evaluation_results)} models evaluated")
        return evaluation_results

    def create_ensemble_prediction(self, X: np.ndarray,
                                 method: str = 'weighted_average') -> np.ndarray:
        """Create ensemble prediction from multiple models."""
        logger.info(f"Creating ensemble prediction using {method}...")

        if not self.trained_models:
            logger.warning("No trained models for ensemble")
            return np.array([])

        # Get predictions from each model
        predictions = []
        weights = []

        for model_type, model_info in self.trained_models.items():
            try:
                model = model_info['model']
                X_use = model_info.get('X_test_scaled', X)  # Simplified - should scale appropriately

                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X_use)[:, 1]
                else:
                    # Convert predictions to probabilities
                    pred = model.predict(X_use)
                    pred_proba = np.where(pred == 1, 0.8, 0.2)  # Simplified

                predictions.append(pred_proba)

                # Weight by model performance (F1 score)
                f1_score = model_info['metrics'].get('f1', 0.5)
                weights.append(max(f1_score, 0.1))  # Ensure minimum weight

            except Exception as e:
                logger.warning(f"Could not get predictions from {model_type}: {e}")
                continue

        if not predictions:
            logger.error("No valid predictions for ensemble")
            return np.array([])

        predictions = np.array(predictions)
        weights = np.array(weights)

        if method == 'weighted_average':
            # Normalize weights
            weights = weights / weights.sum()
            # Weighted average
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
        elif method == 'voting':
            # Majority voting
            binary_preds = (predictions > 0.5).astype(int)
            ensemble_pred = np.mean(binary_preds, axis=0)  # Probability of positive vote
        else:
            # Simple average
            ensemble_pred = np.mean(predictions, axis=0)

        logger.info(f"Ensemble prediction complete: {len(ensemble_pred)} predictions")
        return ensemble_pred

    def predict_championship_probability(self, current_date: datetime = None) -> Dict[str, Any]:
        """Predict championship probabilities for the current season."""
        logger.info("Predicting championship probabilities...")

        if current_date is None:
            current_date = datetime.now()

        # Get current year
        current_year = current_date.year

        # Load most recent data
        self.load_processed_data()
        if self.features is None:
            self.engineer_features()

        # Filter to current season data (most recent races)
        if not self.features.empty and 'year' in self.features.columns:
            current_season_data = self.features[self.features['year'] == current_year].copy()
        else:
            # Fallback to most recent data available
            if not self.features.empty:
                max_year = self.features['year'].max()
                current_season_data = self.features[self.features['year'] == max_year].copy()
            else:
                current_season_data = pd.DataFrame()

        if current_season_data.empty:
            logger.warning("No current season data available for prediction")
            return {}

        # Get latest race completed
        if not current_season_data.empty and 'round' in current_season_data.columns:
            latest_round = current_season_data['round'].max()
            latest_races = current_season_data[current_season_data['round'] == latest_round].copy()
        else:
            latest_races = current_season_data.copy()

        # Make predictions for each driver in the latest race
        if not latest_races.empty:
            # Prepare features for prediction
            X_latest, _, feature_names = self.models.prepare_features(latest_races)

            if len(X_latest) > 0:
                # Get ensemble prediction
                win_probabilities = self.create_ensemble_prediction(X_latest)

                # Add predictions to data
                latest_races = latest_races.copy()
                latest_races['win_probability'] = win_probabilities

                # Get driver information
                driver_info_latest = []
                for _, driver in latest_races.iterrows():
                    driver_id = driver['driver_id']

                    # Get additional driver info from processed data
                    driver_info = {}
                    if 'driver_standings' in self.data and not self.data['driver_standings'].empty:
                        driver_standing = self.data['driver_standings'][
                            self.data['driver_standings']['driver_id'] == driver_id
                        ]
                        if not driver_standing.empty:
                            latest_standing = driver_standing.sort_values('year').iloc[-1]
                            driver_info.update({
                                'championship_points': latest_standing.get('points', 0),
                                'championship_position': latest_standing.get('position', 0),
                                'wins': latest_standing.get('wins', 0)
                            })

                    if 'constructor_standings' in self.data and not self.data['constructor_standings'].empty:
                        const_id = driver.get('constructor_id')
                        if const_id and pd.notna(const_id):
                            const_standing = self.data['constructor_standings'][
                                self.data['constructor_standings']['constructor_id'] == const_id
                            ]
                            if not const_standing.empty:
                                latest_const = const_standing.sort_values('year').iloc[-1]
                                driver_info.update({
                                    'constructor_points': latest_const.get('points', 0),
                                    'constructor_position': latest_const.get('position', 0)
                                })

                    driver_info_latest.append(driver_info)

                # Create championship prediction summary
                championship_prediction = []
                for idx, (_, driver) in enumerate(latest_races.iterrows()):
                    driver_id = driver['driver_id']
                    driver_info = driver_info_latest[idx] if idx < len(driver_info_latest) else {}

                    prediction_entry = {
                        'driver_id': driver_id,
                        'driver_name': f"{driver.get('driver_first_name', '')} {driver.get('driver_last_name', '')}".strip(),
                        'team': driver.get('constructor_name', 'Unknown'),
                        'current_points': driver_info.get('championship_points', 0),
                        'current_position': driver_info.get('championship_position', 0),
                        'win_probability': float(driver['win_probability']) if 'win_probability' in driver.columns else 0.0,
                        'predicted_final_points': 0,  # Would come from simulation
                        'confidence': 0.8  # Placeholder
                    }
                    championship_prediction.append(prediction_entry)

                # Sort by win probability
                championship_prediction.sort(key=lambda x: x['win_probability'], reverse=True)

                # Calculate remaining points available (simplified)
                races_remaining = 0
                if 'races' in self.data and not self.data['races'].empty:
                    current_year_races = self.data['races'][
                        self.data['races']['year'] == current_year
                    ]
                    if not current_year_races.empty:
                        max_round = current_year_races['round'].max()
                        races_remaining = max(0, max_round - latest_round)

                max_points_per_race = 26  # 25 for win + 1 for fastest lap
                max_remaining_points = races_remaining * max_points_per_race

                # Add simulation-based predictions if models are trained
                if self.trained_models:
                    # Run championship simulation
                    sim_results = self.simulator.simulate_championship(
                        latest_races,  # Use current race predictions
                        n_simulations=5000
                    )
                    if sim_results and 'championship_summary' in sim_results:
                        # Update predictions with simulation results
                        sim_summary = sim_results['championship_summary']
                        for pred in championship_prediction:
                            driver_id = pred['driver_id']
                            sim_driver = sim_summary[sim_summary['driver_id'] == driver_id]
                            if not sim_driver.empty:
                                pred['win_probability'] = float(sim_driver.iloc[0]['win_probability'])
                                pred['predicted_final_points'] = float(sim_driver.iloc[0]['mean'])

                # Prepare final results
                results = {
                    'prediction_timestamp': datetime.now(),
                    'current_year': current_year,
                    'latest_race_round': latest_round if not latest_races.empty else 0,
                    'latest_race_name': latest_races.iloc[0].get('race_name', 'Unknown') if not latest_races.empty else 'Unknown',
                    'races_remaining': races_remaining,
                    'max_possible_points': max_remaining_points,
                    'championship_prediction': championship_prediction,
                    'model_performance': {
                        model_type: info['metrics']
                        for model_type, info in self.trained_models.items()
                    }
                }

                logger.info(f"Championship prediction complete for {len(championship_prediction)} drivers")
                return results

        logger.warning("Unable to generate championship prediction")
        return {}

    def explain_prediction_for_driver(self, driver_id: str, race_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate explanation for a specific driver's prediction."""
        logger.info(f"Generating explanation for driver {driver_id}...")

        if not self.trained_models:
            logger.warning("No trained models available for explanation")
            return {}

        # Get the best performing model for explanation
        best_model_type = None
        best_f1 = 0
        for model_type, model_info in self.trained_models.items():
            f1_score = model_info['metrics'].get('f1', 0)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model_type = model_type

        if best_model_type is None:
            best_model_type = list(self.trained_models.keys())[0] if self.trained_models else None

        if best_model_type is None:
            logger.warning("No models available for explanation")
            return {}

        # Prepare data for this driver
        if self.features is not None and not self.features.empty:
            driver_data = self.features[self.features['driver_id'] == driver_id].copy()
            if not driver_data.empty:
                # Get most recent race for this driver
                if 'year' in driver_data.columns and 'round' in driver_data.columns:
                    driver_data = driver_data.sort_values(['year', 'round']).iloc[-1:].copy()

                # Prepare features
                X_driver, _, feature_names = self.models.prepare_features(driver_data)

                if len(X_driver) > 0:
                    # Create SHAP explainer if not exists
                    if best_model_type not in self.explainability.explainers:
                        # Get training data for this model type
                        model_info = self.trained_models[best_model_type]
                        X_train = model_info.get('X_train')
                        if X_train is not None:
                            self.explainability.create_shap_explainer(
                                model_info['model'], best_model_type, X_train, feature_names
                            )

                    # Calculate SHAP values
                    try:
                        self.explainability.calculate_shap_values(best_model_type, X_driver, feature_names)
                    except Exception as e:
                        logger.warning(f"Could not calculate SHAP values: {e}")

                    # Generate explanation
                    explanation = self.explainability.explain_prediction(
                        best_model_type, X_driver, feature_names, instance_index=0
                    )

                    # Add driver context
                    if explanation:
                        explanation['driver_id'] = driver_id
                        explanation['race_context'] = race_context or {}
                        explanation['model_used'] = best_model_type

                    return explanation

        logger.warning(f"Could not generate explanation for driver {driver_id}")
        return {}

    def run_full_pipeline(self, start_year: int = 2015,
                         target_type: str = 'win_race',
                         n_simulations: int = 5000) -> Dict[str, Any]:
        """Run the complete prediction pipeline."""
        logger.info("Starting full F1 championship prediction pipeline...")

        pipeline_results = {
            'start_time': datetime.now(),
            'steps_completed': [],
            'errors': []
        }

        try:
            # Step 1: Initialize data
            logger.info("Step 1: Initializing data...")
            self.initialize_data(start_year=start_year)
            pipeline_results['steps_completed'].append('data_initialization')

            # Step 2: Clean data
            logger.info("Step 2: Cleaning data...")
            self.clean_data()
            pipeline_results['steps_completed'].append('data_cleaning')

            # Step 3: Load processed data
            logger.info("Step 3: Loading processed data...")
            self.load_processed_data()
            pipeline_results['steps_completed'].append('data_loading')

            # Step 4: Engineer features
            logger.info("Step 4: Engineering features...")
            self.engineer_features()
            pipeline_results['steps_completed'].append('feature_engineering')

            # Step 5: Train models
            logger.info("Step 5: Training models...")
            self.train_models(target_type=target_type)
            pipeline_results['steps_completed'].append('model_training')

            # Step 6: Evaluate models
            logger.info("Step 6: Evaluating models...")
            self.evaluate_models()
            pipeline_results['steps_completed'].append('model_evaluation')

            # Step 7: Generate predictions
            logger.info("Step 7: Generating championship predictions...")
            predictions = self.predict_championship_probability()
            pipeline_results['predictions'] = predictions
            pipeline_results['steps_completed'].append('prediction_generation')

            # Step 8: Generate explanations (for top drivers)
            if predictions and 'championship_prediction' in predictions:
                top_drivers = predictions['championship_prediction'][:3]  # Top 3
                explanations = {}
                for driver in top_drivers:
                    driver_id = driver['driver_id']
                    explanation = self.explain_prediction_for_driver(driver_id)
                    explanations[driver_id] = explanation
                pipeline_results['explanations'] = explanations
                pipeline_results['steps_completed'].append('explanation_generation')

            pipeline_results['end_time'] = datetime.now()
            pipeline_results['total_time'] = (pipeline_results['end_time'] - pipeline_results['start_time']).total_seconds()

            logger.info(f"Full pipeline completed in {pipeline_results['total_time']:.2f} seconds")
            logger.info(f"Steps completed: {', '.join(pipeline_results['steps_completed'])}")

        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            pipeline_results['errors'].append(str(e))
            pipeline_results['end_time'] = datetime.now()

        return pipeline_results

def main():
    """Main function for running the prediction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Run F1 Championship Predictor pipeline")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year for historical data")
    parser.add_argument("--target", type=str, default="win_race", help="Target variable to predict")
    parser.add_argument("--simulations", type=int, default=5000, help="Number of Monte Carlo simulations")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training (use existing models)")
    parser.add_argument("--predict-only", action="store_true", help="Only generate predictions, don't train")

    args = parser.parse_args()

    # Initialize predictor
    predictor = F1ChampionshipPredictor()

    if args.predict_only:
        # Only generate predictions
        predictions = predictor.predict_championship_probability()
        print(json.dumps(predictions, indent=2, default=str))
    elif args.skip_training:
        # Load existing models and run prediction
        predictor.load_processed_data()
        predictor.engineer_features()
        # Models would need to be loaded separately
        predictions = predictor.predict_championship_probability()
        print(json.dumps(predictions, indent=2, default=str))
    else:
        # Run full pipeline
        results = predictor.run_full_pipeline(
            start_year=args.start_year,
            target_type=args.target,
            n_simulations=args.simulations
        )
        print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()