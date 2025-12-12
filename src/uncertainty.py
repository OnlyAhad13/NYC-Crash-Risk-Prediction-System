"""
Uncertainty Quantification Module for NYC Crash Risk Prediction

Provides calibrated prediction intervals through:
- Conformal Prediction (distribution-free intervals)
- Quantile Regression
- Prediction interval calibration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PredictionInterval:
    """Container for prediction with uncertainty."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    interval_width: float


class ConformalPredictor:
    """
    Conformal Prediction for distribution-free prediction intervals.
    
    Uses the split conformal method to generate calibrated prediction
    intervals with guaranteed coverage.
    """
    
    def __init__(
        self, 
        model: Any,
        confidence_level: float = 0.90
    ):
        """
        Initialize Conformal Predictor.
        
        Args:
            model: Trained regression model
            confidence_level: Target coverage (e.g., 0.90 for 90% intervals)
        """
        self.model = model
        self.confidence_level = confidence_level
        self.calibration_scores = None
        self.quantile_threshold = None
        
    def calibrate(
        self, 
        X_cal: pd.DataFrame, 
        y_cal: pd.Series
    ) -> float:
        """
        Calibrate the conformal predictor using calibration data.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
            
        Returns:
            Quantile threshold for intervals
        """
        # Get predictions on calibration set
        predictions = self.model.predict(X_cal)
        
        # Calculate nonconformity scores (absolute residuals)
        self.calibration_scores = np.abs(y_cal.values - predictions)
        
        # Find the quantile threshold
        n = len(self.calibration_scores)
        quantile = np.ceil((n + 1) * self.confidence_level) / n
        quantile = min(quantile, 1.0)
        
        self.quantile_threshold = np.quantile(self.calibration_scores, quantile)
        
        return self.quantile_threshold
    
    def predict_intervals(
        self, 
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate prediction intervals.
        
        Args:
            X: Features to predict
            
        Returns:
            Tuple of (point_estimates, lower_bounds, upper_bounds)
        """
        if self.quantile_threshold is None:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")
        
        point_estimates = self.model.predict(X)
        
        lower_bounds = point_estimates - self.quantile_threshold
        upper_bounds = point_estimates + self.quantile_threshold
        
        # Clip to non-negative (count data)
        lower_bounds = np.maximum(lower_bounds, 0)
        upper_bounds = np.maximum(upper_bounds, lower_bounds)
        
        return point_estimates, lower_bounds, upper_bounds
    
    def predict_with_uncertainty(
        self, 
        X: pd.DataFrame
    ) -> List[PredictionInterval]:
        """
        Generate prediction intervals as PredictionInterval objects.
        
        Args:
            X: Features to predict
            
        Returns:
            List of PredictionInterval objects
        """
        points, lowers, uppers = self.predict_intervals(X)
        
        results = []
        for i in range(len(points)):
            results.append(PredictionInterval(
                point_estimate=points[i],
                lower_bound=lowers[i],
                upper_bound=uppers[i],
                confidence_level=self.confidence_level,
                interval_width=uppers[i] - lowers[i]
            ))
        
        return results
    
    def evaluate_coverage(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate coverage and interval width on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict with coverage and width metrics
        """
        points, lowers, uppers = self.predict_intervals(X_test)
        y_vals = y_test.values
        
        # Coverage: fraction of true values within intervals
        covered = (y_vals >= lowers) & (y_vals <= uppers)
        coverage = covered.mean()
        
        # Interval widths
        widths = uppers - lowers
        
        return {
            'empirical_coverage': coverage,
            'target_coverage': self.confidence_level,
            'coverage_gap': coverage - self.confidence_level,
            'mean_interval_width': widths.mean(),
            'median_interval_width': np.median(widths),
            'std_interval_width': widths.std()
        }


class AdaptiveConformalPredictor:
    """
    Adaptive Conformal Prediction with locally-weighted intervals.
    
    Produces tighter intervals for easy predictions and wider
    intervals for harder predictions.
    """
    
    def __init__(
        self, 
        model: Any,
        difficulty_model: Any = None,
        confidence_level: float = 0.90
    ):
        """
        Initialize Adaptive Conformal Predictor.
        
        Args:
            model: Main regression model
            difficulty_model: Model to predict difficulty (optional)
            confidence_level: Target coverage
        """
        self.model = model
        self.difficulty_model = difficulty_model
        self.confidence_level = confidence_level
        self.calibration_data = None
        
    def calibrate(
        self, 
        X_cal: pd.DataFrame, 
        y_cal: pd.Series
    ):
        """
        Calibrate the adaptive predictor.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        predictions = self.model.predict(X_cal)
        residuals = np.abs(y_cal.values - predictions)
        
        # Use predictions as difficulty proxy if no difficulty model
        if self.difficulty_model is None:
            difficulty = np.maximum(predictions, 0.1)  # Avoid division by zero
        else:
            difficulty = self.difficulty_model.predict(X_cal)
        
        # Normalize residuals by difficulty
        normalized_scores = residuals / difficulty
        
        self.calibration_data = {
            'scores': normalized_scores,
            'difficulty': difficulty,
            'quantile': np.quantile(
                normalized_scores, 
                np.ceil((len(normalized_scores) + 1) * self.confidence_level) / len(normalized_scores)
            )
        }
        
        return self
    
    def predict_intervals(
        self, 
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate adaptive prediction intervals.
        
        Args:
            X: Features to predict
            
        Returns:
            Tuple of (point_estimates, lower_bounds, upper_bounds)
        """
        if self.calibration_data is None:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")
        
        predictions = self.model.predict(X)
        
        # Estimate difficulty for new points
        if self.difficulty_model is None:
            difficulty = np.maximum(predictions, 0.1)
        else:
            difficulty = self.difficulty_model.predict(X)
        
        # Interval width scales with difficulty
        margin = self.calibration_data['quantile'] * difficulty
        
        lower_bounds = np.maximum(predictions - margin, 0)
        upper_bounds = predictions + margin
        
        return predictions, lower_bounds, upper_bounds


class QuantileRegressionWrapper:
    """
    Wrapper for quantile regression using gradient boosting.
    
    Trains separate models for different quantiles to get
    prediction intervals.
    """
    
    def __init__(
        self, 
        quantiles: List[float] = [0.1, 0.5, 0.9],
        model_params: Optional[Dict] = None
    ):
        """
        Initialize Quantile Regression.
        
        Args:
            quantiles: List of quantiles to predict
            model_params: Parameters for gradient boosting
        """
        self.quantiles = sorted(quantiles)
        self.model_params = model_params or {}
        self.models: Dict[float, Any] = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit quantile regression models.
        
        Args:
            X: Training features
            y: Training targets
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        for q in self.quantiles:
            print(f"  Training quantile {q:.2f} model...")
            
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 5),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                random_state=42
            )
            
            model.fit(X, y)
            self.models[q] = model
        
        return self
    
    def predict_quantiles(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Predict all quantiles.
        
        Args:
            X: Features to predict
            
        Returns:
            Dict mapping quantile to predictions
        """
        return {q: model.predict(X) for q, model in self.models.items()}
    
    def predict_intervals(
        self, 
        X: pd.DataFrame,
        lower_quantile: float = 0.1,
        upper_quantile: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get prediction intervals from quantile predictions.
        
        Args:
            X: Features to predict
            lower_quantile: Lower bound quantile
            upper_quantile: Upper bound quantile
            
        Returns:
            Tuple of (median, lower, upper)
        """
        predictions = self.predict_quantiles(X)
        
        median = predictions.get(0.5, predictions[self.quantiles[len(self.quantiles)//2]])
        lower = predictions[lower_quantile]
        upper = predictions[upper_quantile]
        
        # Ensure ordering
        lower = np.minimum(lower, median)
        upper = np.maximum(upper, median)
        
        # Clip to non-negative
        lower = np.maximum(lower, 0)
        
        return median, lower, upper


class UncertaintyAnalyzer:
    """
    Comprehensive uncertainty analysis for crash risk predictions.
    """
    
    def __init__(
        self, 
        conformal_predictor: ConformalPredictor,
        quantile_predictor: Optional[QuantileRegressionWrapper] = None
    ):
        self.conformal = conformal_predictor
        self.quantile = quantile_predictor
    
    def analyze_uncertainty_by_feature(
        self, 
        X: pd.DataFrame,
        feature_name: str,
        n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Analyze how uncertainty varies with a feature.
        
        Args:
            X: Features
            feature_name: Feature to analyze
            n_bins: Number of bins
            
        Returns:
            DataFrame with uncertainty analysis
        """
        points, lowers, uppers = self.conformal.predict_intervals(X)
        widths = uppers - lowers
        
        X_analysis = X.copy()
        X_analysis['prediction'] = points
        X_analysis['interval_width'] = widths
        X_analysis['feature_bin'] = pd.cut(X[feature_name], bins=n_bins)
        
        analysis = X_analysis.groupby('feature_bin').agg({
            'prediction': ['mean', 'std'],
            'interval_width': ['mean', 'std', 'count']
        }).round(4)
        
        analysis.columns = [
            'mean_prediction', 'std_prediction',
            'mean_width', 'std_width', 'count'
        ]
        
        return analysis.reset_index()
    
    def get_uncertainty_summary(
        self, 
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive uncertainty summary.
        
        Args:
            X: Features
            y: Optional true values for coverage calculation
            
        Returns:
            Dict with uncertainty metrics
        """
        points, lowers, uppers = self.conformal.predict_intervals(X)
        widths = uppers - lowers
        
        summary = {
            'n_samples': len(X),
            'mean_prediction': points.mean(),
            'std_prediction': points.std(),
            'mean_interval_width': widths.mean(),
            'median_interval_width': np.median(widths),
            'min_interval_width': widths.min(),
            'max_interval_width': widths.max(),
            'confidence_level': self.conformal.confidence_level
        }
        
        if y is not None:
            covered = (y.values >= lowers) & (y.values <= uppers)
            summary['empirical_coverage'] = covered.mean()
            summary['coverage_gap'] = covered.mean() - self.conformal.confidence_level
        
        return summary


def create_uncertainty_pipeline(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    calibration_fraction: float = 0.2,
    confidence_level: float = 0.90
) -> ConformalPredictor:
    """
    Convenience function to create calibrated conformal predictor.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        calibration_fraction: Fraction of data for calibration
        confidence_level: Target coverage
        
    Returns:
        Calibrated ConformalPredictor
    """
    # Split off calibration set
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train, y_train,
        test_size=calibration_fraction,
        random_state=42
    )
    
    # Create and calibrate predictor
    predictor = ConformalPredictor(model, confidence_level)
    predictor.calibrate(X_cal, y_cal)
    
    print(f"Conformal predictor calibrated:")
    print(f"  Calibration samples: {len(X_cal)}")
    print(f"  Confidence level: {confidence_level}")
    print(f"  Quantile threshold: {predictor.quantile_threshold:.4f}")
    
    return predictor


if __name__ == '__main__':
    print("Uncertainty Quantification Module")
    print("Available methods:")
    print("  - ConformalPredictor: Split conformal prediction")
    print("  - AdaptiveConformalPredictor: Locally-weighted intervals")
    print("  - QuantileRegressionWrapper: Gradient boosting quantiles")
