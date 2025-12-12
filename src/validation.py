"""
Validation Module for NYC Crash Risk Prediction

Provides proper evaluation strategies:
- Time-Series Cross-Validation (expanding/sliding window)
- Spatial Cross-Validation (group k-fold by region)
- Custom evaluation metrics
- Learning curves
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Generator, Callable
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CVResult:
    """Container for cross-validation results."""
    fold: int
    train_size: int
    test_size: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: Dict[str, float]


class TimeSeriesValidator:
    """
    Time-Series Cross-Validation with expanding or sliding window.
    
    Ensures no data leakage by always training on past data
    and testing on future data.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        method: str = 'expanding'
    ):
        """
        Initialize Time-Series Validator.
        
        Args:
            n_splits: Number of CV folds
            test_size: Size of test set per fold
            gap: Gap between train and test (for realistic scenarios)
            method: 'expanding' or 'sliding'
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.method = method
        
    def split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        time_column: str = 'hour'
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test splits respecting temporal order.
        
        Args:
            X: Features DataFrame with time column
            y: Target series
            time_column: Name of datetime column
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        # Sort by time
        sort_idx = X[time_column].argsort()
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        if self.method == 'expanding':
            # Expanding window: train set grows, test slides forward
            min_train_size = n_samples // (self.n_splits + 1)
            
            for i in range(self.n_splits):
                test_start = min_train_size + i * test_size + self.gap
                test_end = test_start + test_size
                
                if test_end > n_samples:
                    break
                
                train_idx = sort_idx[:test_start - self.gap]
                test_idx = sort_idx[test_start:test_end]
                
                yield train_idx, test_idx
                
        else:  # sliding
            # Sliding window: fixed window size slides forward
            window_size = n_samples // (self.n_splits + 1) * 2
            
            for i in range(self.n_splits):
                train_start = i * test_size
                train_end = train_start + window_size
                test_start = train_end + self.gap
                test_end = test_start + test_size
                
                if test_end > n_samples:
                    break
                
                train_idx = sort_idx[train_start:train_end]
                test_idx = sort_idx[test_start:test_end]
                
                yield train_idx, test_idx
    
    def cross_validate(
        self,
        model_factory: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
        time_column: str = 'hour',
        metrics: Optional[Dict[str, Callable]] = None
    ) -> Tuple[pd.DataFrame, List[CVResult]]:
        """
        Perform time-series cross-validation.
        
        Args:
            model_factory: Callable that creates a new model instance
            X: Features DataFrame
            y: Target series
            feature_cols: Feature columns to use
            time_column: Datetime column name
            metrics: Dict of metric_name -> metric_function
            
        Returns:
            Tuple of (metrics_df, cv_results)
        """
        if metrics is None:
            metrics = {
                'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error
            }
        
        results = []
        all_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(self.split(X, y, time_column)):
            print(f"  Fold {fold + 1}/{self.n_splits}...")
            
            X_train = X.iloc[train_idx][feature_cols]
            X_test = X.iloc[test_idx][feature_cols]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # Train model
            model = model_factory()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = np.clip(model.predict(X_test), 0, None)
            
            # Calculate metrics
            fold_metrics = {name: func(y_test, y_pred) for name, func in metrics.items()}
            fold_metrics['fold'] = fold + 1
            all_metrics.append(fold_metrics)
            
            # Store result
            results.append(CVResult(
                fold=fold + 1,
                train_size=len(train_idx),
                test_size=len(test_idx),
                train_start=str(X.iloc[train_idx][time_column].min()),
                train_end=str(X.iloc[train_idx][time_column].max()),
                test_start=str(X.iloc[test_idx][time_column].min()),
                test_end=str(X.iloc[test_idx][time_column].max()),
                metrics=fold_metrics
            ))
            
            print(f"    Train: {len(train_idx):,} samples, Test: {len(test_idx):,} samples")
            print(f"    RMSE: {fold_metrics['RMSE']:.4f}, MAE: {fold_metrics['MAE']:.4f}")
        
        metrics_df = pd.DataFrame(all_metrics)
        
        # Add summary stats
        print(f"\n  Mean RMSE: {metrics_df['RMSE'].mean():.4f} (+/- {metrics_df['RMSE'].std():.4f})")
        print(f"  Mean MAE: {metrics_df['MAE'].mean():.4f} (+/- {metrics_df['MAE'].std():.4f})")
        
        return metrics_df, results


class SpatialValidator:
    """
    Spatial Cross-Validation using group k-fold.
    
    Ensures hexagons in test set are not in training set,
    testing generalization to new locations.
    """
    
    def __init__(self, n_splits: int = 5):
        """
        Initialize Spatial Validator.
        
        Args:
            n_splits: Number of CV folds
        """
        self.n_splits = n_splits
        self.gkf = GroupKFold(n_splits=n_splits)
        
    def split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        groups: pd.Series
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate spatially-separated train/test splits.
        
        Args:
            X: Features DataFrame
            y: Target series
            groups: Spatial group identifiers (e.g., h3_index)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        return self.gkf.split(X, y, groups)
    
    def cross_validate(
        self,
        model_factory: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        feature_cols: List[str],
        metrics: Optional[Dict[str, Callable]] = None
    ) -> pd.DataFrame:
        """
        Perform spatial cross-validation.
        
        Args:
            model_factory: Callable that creates a new model instance
            X: Features DataFrame
            y: Target series
            groups: Spatial groups
            feature_cols: Feature columns
            metrics: Metric functions
            
        Returns:
            DataFrame with metrics per fold
        """
        if metrics is None:
            metrics = {
                'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error
            }
        
        all_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(self.split(X, y, groups)):
            print(f"  Spatial Fold {fold + 1}/{self.n_splits}...")
            
            X_train = X.iloc[train_idx][feature_cols]
            X_test = X.iloc[test_idx][feature_cols]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # Check spatial separation
            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])
            overlap = train_groups & test_groups
            
            if overlap:
                print(f"    Warning: {len(overlap)} overlapping groups")
            
            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = np.clip(model.predict(X_test), 0, None)
            
            fold_metrics = {name: func(y_test, y_pred) for name, func in metrics.items()}
            fold_metrics['fold'] = fold + 1
            fold_metrics['n_train_groups'] = len(train_groups)
            fold_metrics['n_test_groups'] = len(test_groups)
            all_metrics.append(fold_metrics)
            
            print(f"    Train hexagons: {len(train_groups)}, Test hexagons: {len(test_groups)}")
            print(f"    RMSE: {fold_metrics['RMSE']:.4f}")
        
        return pd.DataFrame(all_metrics)


class CustomMetrics:
    """
    Custom evaluation metrics for crash risk prediction.
    """
    
    @staticmethod
    def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Poisson deviance for count data.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Poisson deviance
        """
        y_pred = np.maximum(y_pred, 1e-10)  # Avoid log(0)
        y_true = np.maximum(y_true, 0)
        
        deviance = 2 * np.sum(
            y_true * np.log(np.maximum(y_true, 1e-10) / y_pred) - (y_true - y_pred)
        )
        
        return deviance / len(y_true)
    
    @staticmethod
    def weighted_rmse(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Weighted RMSE (e.g., weight by severity or importance).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            weights: Sample weights
            
        Returns:
            Weighted RMSE
        """
        if weights is None:
            # Weight by true value (penalize errors on high-accident areas more)
            weights = np.maximum(y_true, 1)
        
        squared_errors = (y_true - y_pred) ** 2
        weighted_mse = np.sum(weights * squared_errors) / np.sum(weights)
        
        return np.sqrt(weighted_mse)
    
    @staticmethod
    def high_risk_recall(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        threshold_percentile: float = 90
    ) -> float:
        """
        Recall for high-risk predictions.
        
        What fraction of actual high-risk events were predicted as high-risk?
        
        Args:
            y_true: True values
            y_pred: Predicted values
            threshold_percentile: Percentile to define "high risk"
            
        Returns:
            Recall for high-risk events
        """
        threshold = np.percentile(y_true, threshold_percentile)
        
        actual_high = y_true >= threshold
        predicted_high = y_pred >= np.percentile(y_pred, threshold_percentile)
        
        if actual_high.sum() == 0:
            return 1.0
        
        return (actual_high & predicted_high).sum() / actual_high.sum()
    
    @staticmethod
    def calibration_error(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error for predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            n_bins: Number of bins
            
        Returns:
            ECE score
        """
        bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1  # Include max value
        
        ece = 0
        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_pred_mean = y_pred[mask].mean()
                bin_true_mean = y_true[mask].mean()
                ece += mask.sum() * abs(bin_pred_mean - bin_true_mean)
        
        return ece / len(y_true)


def plot_learning_curve(
    model_factory: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    train_sizes: Optional[List[float]] = None,
    cv: int = 3,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Plot learning curve showing train vs validation error.
    
    Args:
        model_factory: Callable creating model instances
        X: Features
        y: Target
        feature_cols: Feature columns
        train_sizes: List of training set fractions to evaluate
        cv: Number of CV folds
        save_path: Optional path to save plot
        
    Returns:
        DataFrame with learning curve data
    """
    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    
    for train_frac in train_sizes:
        n_samples = int(len(X) * train_frac)
        
        train_scores = []
        val_scores = []
        
        for fold in range(cv):
            # Random subset for this fraction
            np.random.seed(fold)
            indices = np.random.permutation(len(X))[:n_samples]
            
            # Split into train/val
            split = int(0.8 * len(indices))
            train_idx = indices[:split]
            val_idx = indices[split:]
            
            X_train = X.iloc[train_idx][feature_cols]
            X_val = X.iloc[val_idx][feature_cols]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
            
            model = model_factory()
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_scores.append(np.sqrt(mean_squared_error(y_train, train_pred)))
            val_scores.append(np.sqrt(mean_squared_error(y_val, val_pred)))
        
        results.append({
            'train_fraction': train_frac,
            'n_samples': n_samples,
            'train_rmse_mean': np.mean(train_scores),
            'train_rmse_std': np.std(train_scores),
            'val_rmse_mean': np.mean(val_scores),
            'val_rmse_std': np.std(val_scores)
        })
    
    df = pd.DataFrame(results)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.fill_between(
        df['n_samples'],
        df['train_rmse_mean'] - df['train_rmse_std'],
        df['train_rmse_mean'] + df['train_rmse_std'],
        alpha=0.2, color='blue'
    )
    plt.fill_between(
        df['n_samples'],
        df['val_rmse_mean'] - df['val_rmse_std'],
        df['val_rmse_mean'] + df['val_rmse_std'],
        alpha=0.2, color='orange'
    )
    plt.plot(df['n_samples'], df['train_rmse_mean'], 'o-', color='blue', label='Training')
    plt.plot(df['n_samples'], df['val_rmse_mean'], 'o-', color='orange', label='Validation')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved learning curve to {save_path}")
    
    plt.close()
    
    return df


if __name__ == '__main__':
    print("Validation Module")
    print("Available validators:")
    print("  - TimeSeriesValidator: Temporal CV with expanding/sliding window")
    print("  - SpatialValidator: Group K-Fold by location")
    print("  - CustomMetrics: Poisson deviance, weighted RMSE, high-risk recall")
