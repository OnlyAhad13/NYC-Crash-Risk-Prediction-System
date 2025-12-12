"""
SHAP Explainability Module for NYC Crash Risk Prediction

Provides model interpretability through:
- Global feature importance (SHAP summary plots)
- Local explanations (waterfall, force plots)
- Feature interaction analysis
- What-if counterfactual analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class SHAPExplainer:
    """
    SHAP-based model explainability for crash risk prediction.
    
    Supports TreeExplainer (XGBoost, LightGBM, CatBoost, RF) and
    Kernel/Permutation explainers for other models.
    """
    
    def __init__(self, model: Any, feature_names: List[str], model_type: str = 'tree'):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            feature_names: List of feature column names
            model_type: 'tree' for gradient boosting, 'kernel' for others
        """
        if not HAS_SHAP:
            raise ImportError("SHAP not installed. Run: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
    def fit(self, X_background: pd.DataFrame, max_samples: int = 1000):
        """
        Fit the SHAP explainer using background data.
        
        Args:
            X_background: Background dataset for SHAP calculations
            max_samples: Maximum samples to use (for efficiency)
        """
        if len(X_background) > max_samples:
            X_background = X_background.sample(n=max_samples, random_state=42)
        
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.Explainer(self.model, X_background)
        
        self.expected_value = self.explainer.expected_value
        if isinstance(self.expected_value, np.ndarray):
            self.expected_value = self.expected_value[0]
        
        return self
    
    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for given samples.
        
        Args:
            X: Features to explain
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        shap_values = self.explainer.shap_values(X)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        self.shap_values = shap_values
        return shap_values
    
    def get_global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get global feature importance based on mean |SHAP values|.
        
        Args:
            X: Dataset to calculate importance over
            
        Returns:
            DataFrame with feature importance
        """
        shap_values = self.explain(X)
        
        importance = np.abs(shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'importance_pct': importance / importance.sum() * 100
        }).sort_values('importance', ascending=False)
        
        return df
    
    def explain_single(self, X_single: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            X_single: Single row DataFrame
            
        Returns:
            Dictionary with explanation details
        """
        if len(X_single) != 1:
            X_single = X_single.iloc[[0]]
        
        shap_values = self.explain(X_single)[0]
        prediction = self.model.predict(X_single)[0]
        
        # Feature contributions
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_single.values[0],
            'shap_value': shap_values,
            'abs_shap': np.abs(shap_values)
        }).sort_values('abs_shap', ascending=False)
        
        return {
            'prediction': prediction,
            'base_value': self.expected_value,
            'contributions': contributions,
            'top_positive': contributions[contributions['shap_value'] > 0].head(5),
            'top_negative': contributions[contributions['shap_value'] < 0].head(5)
        }
    
    def what_if_analysis(
        self, 
        X_original: pd.DataFrame,
        feature_changes: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform what-if counterfactual analysis.
        
        Args:
            X_original: Original feature values (single row)
            feature_changes: Dict of {feature_name: new_value}
            
        Returns:
            Comparison of original vs modified predictions
        """
        if len(X_original) != 1:
            X_original = X_original.iloc[[0]]
        
        X_modified = X_original.copy()
        for feature, value in feature_changes.items():
            if feature in X_modified.columns:
                X_modified[feature] = value
        
        original_pred = self.model.predict(X_original)[0]
        modified_pred = self.model.predict(X_modified)[0]
        
        original_shap = self.explain(X_original)[0]
        modified_shap = self.explain(X_modified)[0]
        
        changes = []
        for feature, new_value in feature_changes.items():
            if feature in self.feature_names:
                idx = self.feature_names.index(feature)
                changes.append({
                    'feature': feature,
                    'original_value': X_original[feature].values[0],
                    'new_value': new_value,
                    'original_shap': original_shap[idx],
                    'new_shap': modified_shap[idx],
                    'shap_change': modified_shap[idx] - original_shap[idx]
                })
        
        return {
            'original_prediction': original_pred,
            'modified_prediction': modified_pred,
            'prediction_change': modified_pred - original_pred,
            'prediction_change_pct': (modified_pred - original_pred) / max(original_pred, 0.001) * 100,
            'feature_changes': pd.DataFrame(changes)
        }
    
    def plot_summary(
        self, 
        X: pd.DataFrame, 
        max_display: int = 15,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP summary plot (beeswarm).
        
        Args:
            X: Features dataset
            max_display: Maximum features to display
            save_path: Optional path to save plot
        """
        shap_values = self.explain(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved summary plot to {save_path}")
        
        plt.close()
        return save_path
    
    def plot_bar_importance(
        self, 
        X: pd.DataFrame, 
        max_display: int = 15,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP bar importance plot.
        
        Args:
            X: Features dataset
            max_display: Maximum features to display
            save_path: Optional path to save plot
        """
        shap_values = self.explain(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type='bar',
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved bar plot to {save_path}")
        
        plt.close()
        return save_path
    
    def plot_waterfall(
        self, 
        X_single: pd.DataFrame, 
        max_display: int = 15,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP waterfall plot for single prediction.
        
        Args:
            X_single: Single row DataFrame
            max_display: Maximum features to display
            save_path: Optional path to save plot
        """
        if len(X_single) != 1:
            X_single = X_single.iloc[[0]]
        
        shap_values = self.explain(X_single)
        
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.expected_value,
                data=X_single.values[0],
                feature_names=self.feature_names
            ),
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved waterfall plot to {save_path}")
        
        plt.close()
        return save_path
    
    def get_feature_interactions(
        self, 
        X: pd.DataFrame,
        feature1: str,
        feature2: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get SHAP interaction values for feature pairs.
        
        Args:
            X: Features dataset
            feature1: First feature name
            feature2: Second feature (optional, auto-selected if None)
            
        Returns:
            Tuple of (shap_values, feature2_name)
        """
        shap_values = self.explain(X)
        
        idx1 = self.feature_names.index(feature1)
        
        if feature2 is None:
            # Auto-select most interacting feature
            correlations = []
            for i, f in enumerate(self.feature_names):
                if i != idx1:
                    corr = np.corrcoef(X[feature1], shap_values[:, idx1])[0, 1]
                    correlations.append((f, abs(corr)))
            correlations.sort(key=lambda x: x[1], reverse=True)
            feature2 = correlations[0][0]
        
        return shap_values[:, idx1], feature2


class ExplanationReport:
    """
    Generate comprehensive explanation reports for the model.
    """
    
    def __init__(self, explainer: SHAPExplainer):
        self.explainer = explainer
    
    def generate_global_report(
        self, 
        X: pd.DataFrame, 
        output_dir: str
    ) -> Dict[str, str]:
        """
        Generate complete global explanation report.
        
        Args:
            X: Dataset to explain
            output_dir: Directory to save outputs
            
        Returns:
            Dict of output file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        outputs = {}
        
        # 1. Feature importance table
        importance = self.explainer.get_global_importance(X)
        importance_path = os.path.join(output_dir, 'shap_importance.csv')
        importance.to_csv(importance_path, index=False)
        outputs['importance_table'] = importance_path
        
        # 2. Summary plot
        summary_path = os.path.join(output_dir, 'shap_summary.png')
        self.explainer.plot_summary(X, save_path=summary_path)
        outputs['summary_plot'] = summary_path
        
        # 3. Bar importance plot
        bar_path = os.path.join(output_dir, 'shap_bar_importance.png')
        self.explainer.plot_bar_importance(X, save_path=bar_path)
        outputs['bar_plot'] = bar_path
        
        print(f"Global report saved to {output_dir}")
        return outputs
    
    def explain_high_risk_samples(
        self, 
        X: pd.DataFrame, 
        predictions: np.ndarray,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Explain the top N highest risk predictions.
        
        Args:
            X: Features dataset
            predictions: Model predictions
            top_n: Number of samples to explain
            
        Returns:
            List of explanation dicts
        """
        # Get indices of highest predictions
        top_indices = np.argsort(predictions)[-top_n:][::-1]
        
        explanations = []
        for idx in top_indices:
            X_sample = X.iloc[[idx]]
            explanation = self.explainer.explain_single(X_sample)
            explanation['sample_idx'] = idx
            explanation['rank'] = len(explanations) + 1
            explanations.append(explanation)
        
        return explanations


def create_explainer_for_dashboard(
    model: Any,
    X_background: pd.DataFrame,
    feature_names: List[str]
) -> SHAPExplainer:
    """
    Convenience function to create and fit a SHAP explainer for dashboard use.
    
    Args:
        model: Trained model
        X_background: Background dataset (sample)
        feature_names: Feature column names
        
    Returns:
        Fitted SHAPExplainer
    """
    explainer = SHAPExplainer(model, feature_names, model_type='tree')
    explainer.fit(X_background, max_samples=500)
    return explainer


if __name__ == '__main__':
    print("SHAP Explainability Module")
    print("SHAP available:", HAS_SHAP)
    if HAS_SHAP:
        print("SHAP version:", shap.__version__)
