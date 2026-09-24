"""
Explainability module for F1 Championship Predictor.
Handles model interpretation and feature importance analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1Explainability:
    """Handles explainability of F1 prediction models."""

    def __init__(self):
        self.explainers = {}
        self.shap_values = {}

    def create_shap_explainer(self, model: Any, model_type: str,
                            X_train: np.ndarray = None,
                            feature_names: List[str] = None) -> Any:
        """Create a SHAP explainer for the given model."""
        logger.info(f"Creating SHAP explainer for {model_type}...")

        try:
            if model_type in ['tree', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost']:
                # Tree-based models
                explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                # Linear models
                if X_train is not None:
                    explainer = shap.LinearExplainer(model, X_train)
                else:
                    explainer = shap.LinearExplainer(model)
            else:
                # Kernel explainer (slower but works with any model)
                if X_train is not None:
                    # Use a sample for efficiency
                    sample_size = min(100, len(X_train))
                    X_sample = shap.sample(X_train, sample_size)
                    explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, 'predict_proba') else model.predict, X_sample)
                else:
                    raise ValueError("X_train required for KernelExplainer")

            self.explainers[model_type] = explainer
            logger.info(f"SHAP explainer created for {model_type}")
            return explainer

        except Exception as e:
            logger.error(f"Error creating SHAP explainer for {model_type}: {e}")
            # Fallback to basic feature importance
            return None

    def calculate_shap_values(self, model_type: str, X: np.ndarray,
                            feature_names: List[str] = None) -> np.ndarray:
        """Calculate SHAP values for given data."""
        if model_type not in self.explainers:
            raise ValueError(f"No explainer found for {model_type}. Create explainer first.")

        explainer = self.explainers[model_type]

        try:
            # Calculate SHAP values
            if hasattr(explainer, 'shap_values'):
                shap_values = explainer.shap_values(X)
                # For binary classification, shap_values is often a list [class0, class1]
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]  # Use positive class
                else:
                    shap_values = explainer(X)

            self.shap_values[model_type] = shap_values
            logger.info(f"SHAP values calculated for {model_type}: shape {shap_values.shape}")
            return shap_values

        except Exception as e:
            logger.error(f"Error calculating SHAP values for {model_type}: {e}")
            return np.array([])

    def get_feature_importance_shap(self, model_type: str,
                                  feature_names: List[str] = None) -> pd.DataFrame:
        """Get feature importance based on SHAP values."""
        if model_type not in self.shap_values:
            raise ValueError(f"No SHAP values found for {model_type}. Calculate SHAP values first.")

        shap_values = self.shap_values[model_type]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        logger.info(f"SHAP feature importance calculated for {model_type}")
        return importance_df

    def get_shap_summary_plot(self, model_type: str, X: np.ndarray,
                            feature_names: List[str] = None,
                            max_display: int = 20,
                            save_path: str = None) -> None:
        """Create and save SHAP summary plot."""
        if model_type not in self.shap_values:
            # Calculate SHAP values first
            self.calculate_shap_values(model_type, X, feature_names)

        if model_type not in self.shap_values:
            logger.error(f"No SHAP values available for {model_type}")
            return

        shap_values = self.shap_values[model_type]

        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
            plt.title(f"SHAP Summary Plot - {model_type}")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")

    def get_shap_dependence_plot(self, model_type: str, X: np.ndarray,
                               feature_name: str, feature_names: List[str] = None,
                               interaction_index: str = None,
                               save_path: str = None) -> None:
        """Create and save SHAP dependence plot for a specific feature."""
        if model_type not in self.explainers:
            self.calculate_shap_values(model_type, X, feature_names)

        if model_type not in self.explainers:
            logger.error(f"No SHAP values available for {model_type}")
            return

        shap_values = self.shap_values[model_type]

        try:
            # Get feature index
            if feature_names is not None and feature_name in feature_names:
                feature_idx = feature_names.index(feature_name)
            else:
                # Assume feature names match column order
                feature_idx = list(feature_names).index(feature_name) if feature_names else 0

            plt.figure(figsize=(10, 8))
            shap.dependence_plot(
                feature_idx, shap_values, X,
                feature_names=feature_names,
                interaction_index=interaction_index,
                show=False
            )
            plt.title(f"SHAP Dependence Plot - {feature_name} ({model_type})")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP dependence plot saved to {save_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot: {e}")

    def get_feature_importance_permutation(self, model: Any, X: np.ndarray, y: np.ndarray,
                                         feature_names: List[str] = None,
                                         n_repeats: int = 5,
                                         scoring: str = 'accuracy') -> pd.DataFrame:
        """Calculate feature importance using permutation method."""
        from sklearn.inspection import permutation_importance

        logger.info("Calculating permutation feature importance...")

        try:
            if scoring == 'accuracy':
                from sklearn.metrics import accuracy_score
                scorer = lambda estimator, X_test, y_test: accuracy_score(y_test, estimator.predict(X_test))
            elif scoring == 'f1':
                from sklearn.metrics import f1_score
                scorer = lambda estimator, X_test, y_test: f1_score(y_test, estimator.predict(X_test))
            else:
                scorer = scoring  # Assume it's a valid scorer string or callable

            result = permutation_importance(
                model, X, y,
                n_repeats=n_repeats,
                random_state=42,
                scoring=scorer
            )

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance', ascending=False)

            logger.info(f"Permutation feature importance complete")
            return importance_df

        except Exception as e:
            logger.error(f"Error calculating permutation importance: {e}")
            return pd.DataFrame()

    def explain_prediction(self, model_type: str, X_instance: np.ndarray,
                         feature_names: List[str] = None,
                         instance_index: int = 0) -> Dict[str, Any]:
        """Explain a single prediction using SHAP values.
        """
        if model_type not in self.shap_values:
            logger.error(f"No SHAP values available for {model_type}")
            return {}

        shap_values = self.shap_values[model_type]

        if instance_index >= shap_values.shape[0]:
            logger.error(f"Instance index {instance_index} out of range")
            return {}

        # Get SHAP values for this instance
        instance_shap = shap_values[instance_index]

        # Get feature values for this instance (if available)
        feature_values = None
        # In a full implementation, you'd pass the original X data

        # Create explanation dictionary
        explanation = {
            'expected_value': getattr(self.explainers[model_type], 'expected_value', 0),
            'shap_values': instance_shap.tolist(),
            'feature_names': feature_names or [f"feature_{i}" for i in range(len(instance_shap))],
            'prediction_contribution': instance_shap.sum()
        }

        # Sort features by absolute SHAP value (most influential first)
        if feature_names is not None:
            feature_contributions = list(zip(feature_names, instance_shap))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            explanation['top_positive_features'] = [
                {'feature': feat, 'contribution': float(contrib)}
                for feat, contrib in feature_contributions if contrib > 0
            ][:5]
            explanation['top_negative_features'] = [
                {'feature': feat, 'contribution': float(contrib)}
                for feat, contrib in feature_contributions if contrib < 0
            ][:5]

        logger.info(f"Prediction explanation generated for instance {instance_index}")
        return explanation

    def create_feature_interaction_report(self, model_type: str, X: np.ndarray,
                                        feature_names: List[str] = None,
                                        max_interactions: int = 10) -> Dict[str, Any]:
        """Create report on feature interactions (simplified)."""
        if model_type not in self.shap_values:
            self.calculate_shap_values(model_type, X, feature_names)

        if model_type not in self.shap_values:
            return {}

        shap_values = self.shap_values[model_type]

        # Simple interaction detection: look for features where SHAP values correlate
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

        interactions = []
        n_features = len(feature_names)

        # Calculate correlation matrix of SHAP values
        try:
            shap_corr = np.corrcoef(shap_values.T)  # Features x features

            # Find strongest interactions (high absolute correlation, excluding diagonal)
            for i in range(n_features):
                for j in range(i+1, n_features):
                    corr = shap_corr[i, j]
                    if abs(corr) > 0.3:  # Threshold for interesting interaction
                        interactions.append({
                            'feature_1': feature_names[i],
                            'feature_2': feature_names[j],
                            'interaction_strength': float(corr),
                            'interaction_type': 'positive' if corr > 0 else 'negative'
                        })

            # Sort by absolute interaction strength
            interactions.sort(key=lambda x: abs(x['interaction_strength']), reverse=True)
            interactions = interactions[:max_interactions]

        except Exception as e:
            logger.warning(f"Could not calculate feature interactions: {e}")
            interactions = []

        report = {
            'model_type': model_type,
            'n_features_analyzed': n_features,
            'interactions_found': len(interactions),
            'top_interactions': interactions,
            'shap_value_stats': {
                'mean_abs_shap': np.mean(np.abs(shap_values)),
                'std_shap': np.std(shap_values),
                'max_shap': np.max(shap_values),
                'min_shap': np.min(shap_values)
            }
        }

        logger.info(f"Feature interaction report generated: {len(interactions)} interactions found")
        return report

    def generate_explanation_text(self, explanation: Dict[str, Any],
                                driver_name: str = "Driver") -> str:
        """Generate human-readable explanation text."""
        if not explanation:
            return "No explanation available."

        text_parts = [f"Explanation for {driver_name}'s prediction:"]

        # Expected value
        expected_value = explanation.get('expected_value', 0)
        text_parts.append(f"Baseline prediction: {expected_value:.3f}")

        # Top positive contributions
        top_pos = explanation.get('top_positive_features', [])
        if top_pos:
            text_parts.append("\nTop positive influences:")
            for item in top_pos[:3]:
                text_parts.append(f"  + {item['feature']}: {item['contribution']:.3f}")

        # Top negative contributions
        top_neg = explanation.get('top_negative_features', [])
        if top_neg:
            text_parts.append("\nTop negative influences:")
            for item in top_neg[:3]:
                text_parts.append(f"  - {item['feature']}: {item['contribution']:.3f}")

        # Overall prediction
        pred_contribution = explanation.get('prediction_contribution', 0)
        final_prediction = expected_value + pred_contribution
        text_parts.append(f"\nFinal prediction contribution: {pred_contribution:.3f}")
        text_parts.append(f"Final predicted value: {final_prediction:.3f}")

        return "\n".join(text_parts)

def main():
    """Main function for testing explainability."""
    print("F1 Explainability module ready")

if __name__ == "__main__":
    main()