"""
Evaluation module for F1 Championship Predictor.
Handles model validation and performance assessment.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1Evaluator:
    """Handles evaluation of F1 prediction models."""

    def __init__(self):
        self.evaluation_results = {}

    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None,
                              model_name: str = "model") -> Dict[str, float]:
        """Evaluate classification model performance."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate probability-based metrics: {e}")
                metrics['roc_auc'] = 0.0
                metrics['brier_score'] = 0.0
                metrics['log_loss'] = 0.0

        self.evaluation_results[model_name] = metrics
        logger.info(f"Evaluated {model_name}: Accuracy={metrics['accuracy']:.3d}, F1={metrics['f1']:.3f}")
        return metrics

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = "model") -> Dict[str, float]:
        """Evaluate regression model performance."""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }

        self.evaluation_results[model_name] = metrics
        logger.info(f"Evaluated {model_name}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")
        return metrics

    def evaluate_ranking(self, y_true: np.ndarray, y_pred: np.ndarray,
                        model_name: str = "model") -> Dict[str, float]:
        """Evaluate ranking performance (for position prediction)."""
        from sklearn.metrics import ndcg_score

        # For ranking, we need to reshape data appropriately
        # This is a simplified version - in practice, you'd group by race
        metrics = {}

        try:
            # Calculate Spearman correlation as a ranking metric
            from scipy.stats import spearmanr
            if len(y_true) > 1 and len(y_pred) > 1:
                corr, _ = spearmanr(y_true, y_pred)
                metrics['spearman_correlation'] = corr if not np.isnan(corr) else 0.0
            else:
                metrics['spearman_correlation'] = 0.0

            # Calculate Kendall's tau
            from scipy.stats import kendalltau
            if len(y_true) > 1 and len(y_pred) > 1:
                tau, _ = kendalltau(y_true, y_pred)
                metrics['kendalls_tau'] = tau if not np.isnan(tau) else 0.0
            else:
                metrics['kendalls_tau'] = 0.0

        except ImportError:
            logger.warning("Scipy not available for ranking metrics")
            metrics['spearman_correlation'] = 0.0
            metrics['kendalls_tau'] = 0.0

        self.evaluation_results[model_name] = metrics
        return metrics

    def chronological_validation(self, df: pd.DataFrame, model_trainer: Any,
                               feature_preparer: Any, target_column: str,
                               start_year: int = 2015) -> Dict[str, List[float]]:
        """Perform chronological/walk-forward validation."""
        logger.info("Starting chronological validation...")

        if df.empty:
            return {}

        # Get unique years and sort them
        years = sorted(df['year'].unique())
        years = [y for y in years if y >= start_year]

        if len(years) < 2:
            logger.warning("Not enough years for chronological validation")
            return {}

        # Store results for each fold
        fold_results = []

        # Iterate through years, expanding training set
        for i in range(1, len(years)):
            train_years = years[:i]
            test_year = years[i]

            logger.info(f"Training on years {train_years}, testing on {test_year}")

            # Split data
            train_mask = df['year'].isin(train_years)
            test_mask = df['year'] == test_year

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            if train_df.empty or test_df.empty:
                logger.warning(f"Empty train or test set for year {test_year}")
                continue

            # Prepare features
            try:
                X_train, y_train, feature_cols = feature_preparer(train_df, target_column)
                X_test, y_test, _ = feature_preparer(test_df, target_column)

                if len(X_train) == 0 or len(X_test) == 0:
                    logger.warning(f"Empty feature sets for year {test_year}")
                    continue

                # Train model
                model = model_trainer(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Evaluate
                metrics = self.evaluate_classification(y_test, y_pred, y_pred_proba, f"fold_{test_year}")
                metrics['test_year'] = test_year
                metrics['train_size'] = len(X_train)
                metrics['test_size'] = len(X_test)
                fold_results.append(metrics)

            except Exception as e:
                logger.error(f"Error in fold for year {test_year}: {e}")
                continue

        # Calculate average metrics across folds
        if fold_results:
            avg_metrics = {}
            numeric_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score', 'log_loss']

            for metric in numeric_metrics:
                values = [fold[metric] for fold in fold_results if metric in fold and not np.isnan(fold[metric])]
                if values:
                    avg_metrics[f'avg_{metric}'] = np.mean(values)
                    avg_metrics[f'std_{metric}'] = np.std(values)

            avg_metrics['folds'] = len(fold_results)
            avg_metrics['fold_results'] = fold_results

            logger.info(f"Chronological validation complete: {len(fold_results)} folds")
            return avg_metrics
        else:
            logger.warning("No valid folds for chronological validation")
            return {}

    def evaluate_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                           n_bins: int = 10) -> Dict[str, Any]:
        """Evaluate probability calibration."""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins
            )

            # Calculate calibration error (Expected Calibration Error)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            calibration_data = {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist(),
                'expected_calibration_error': ece,
                'bin_boundaries': bin_boundaries.tolist()
            }

            logger.info(f"Calibration evaluation complete. ECE: {ece:.4f}")
            return calibration_data

        except Exception as e:
            logger.error(f"Error in calibration evaluation: {e}")
            return {'error': str(e)}

    def plot_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                        model_name: str = "model", save_path: str = None) -> None:
        """Plot calibration curve."""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )

            plt.figure(figsize=(8, 6))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name}")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.ylabel("Fraction of positives")
            plt.xlabel("Mean predicted probability")
            plt.title(f"Calibration curve for {model_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Calibration plot saved to {save_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.error(f"Error plotting calibration: {e}")

    def plot_feature_importance(self, model: Any, feature_names: List[str],
                              model_name: str = "model", top_n: int = 20,
                              save_path: str = None) -> None:
        """Plot feature importance for tree-based models."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                logger.warning(f"Model {model_name} doesn't have feature_importances_ or coef_ attribute")
                return

            # Create DataFrame for sorting
            feat_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(feat_importance)), feat_importance['importance'])
            plt.yticks(range(len(feat_importance)), feat_importance['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance for {model_name}')
            plt.gca().invert_yaxis()  # Highest importance on top
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")

    def save_evaluation_results(self, filepath: str = "evaluation_results.json") -> None:
        """Save evaluation results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        results_serializable = convert_types(self.evaluation_results)

        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Evaluation results saved to {filepath}")

    def load_evaluation_results(self, filepath: str = "evaluation_results.json") -> Dict:
        """Load evaluation results from JSON file."""
        if not os.path.exists(filepath):
            logger.warning(f"Evaluation results file not found: {filepath}")
            return {}

        with open(filepath, 'r') as f:
            results = json.load(f)

        self.evaluation_results = results
        logger.info(f"Evaluation results loaded from {filepath}")
        return results

def compare_models(evaluation_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Compare multiple models based on evaluation results."""
    if not evaluation_results:
        return pd.DataFrame()

    # Create DataFrame for easy comparison
    comparison_df = pd.DataFrame(evaluation_results).T

    # Sort by a primary metric (e.g., F1 score or accuracy)
    sort_columns = ['f1', 'accuracy', 'roc_auc']
    sort_column = None
    for col in sort_columns:
        if col in comparison_df.columns:
            sort_column = col
            break

    if sort_column:
        comparison_df = comparison_df.sort_values(sort_column, ascending=False)

    logger.info(f"Compared {len(evaluation_results)} models")
    return comparison_df

def main():
    """Main function for testing evaluation."""
    print("F1 Evaluation module ready")

if __name__ == "__main__":
    main()