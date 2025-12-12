"""
Advanced Models Module for NYC Crash Risk Prediction

Provides a unified interface for multiple ML models:
- XGBoost (Poisson regression)
- LightGBM (Poisson regression)
- CatBoost (Poisson regression)
- Quantile Regression Forests
- Stacking Ensemble
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import joblib

# Try importing optional dependencies
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


@dataclass
class ModelResult:
    """Container for model training results."""
    name: str
    model: Any
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    metrics: Dict[str, float]
    feature_importance: Optional[pd.DataFrame] = None


class QuantileRandomForest(BaseEstimator, RegressorMixin):
    """
    Quantile Random Forest for prediction intervals.
    
    Fits a Random Forest and uses the individual tree predictions
    to estimate prediction quantiles.
    """
    
    def __init__(
        self, 
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        n_jobs: int = -1,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.quantiles = quantiles
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.rf = None
        
    def fit(self, X, y):
        """Fit the Quantile Random Forest."""
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.rf.fit(X, y)
        return self
    
    def predict(self, X):
        """Return median prediction."""
        return self.predict_quantile(X, 0.5)
    
    def predict_quantile(self, X, quantile: float) -> np.ndarray:
        """Predict a specific quantile."""
        if self.rf is None:
            raise ValueError("Model not fitted yet.")
        
        # Get predictions from all trees
        all_predictions = np.array([
            tree.predict(X) for tree in self.rf.estimators_
        ])
        
        # Return the quantile across trees
        return np.percentile(all_predictions, quantile * 100, axis=0)
    
    def predict_intervals(self, X) -> Dict[float, np.ndarray]:
        """Predict all configured quantiles."""
        return {q: self.predict_quantile(X, q) for q in self.quantiles}


class ModelFactory:
    """Factory for creating different model types."""
    
    @staticmethod
    def create_xgboost(params: Optional[Dict] = None) -> xgb.XGBRegressor:
        """Create XGBoost regressor with Poisson objective."""
        default_params = {
            'objective': 'count:poisson',
            'n_estimators': 200,
            'max_depth': 7,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'random_state': 42,
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        return xgb.XGBRegressor(**default_params)
    
    @staticmethod
    def create_lightgbm(params: Optional[Dict] = None):
        """Create LightGBM regressor with Poisson objective."""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        default_params = {
            'objective': 'poisson',
            'n_estimators': 200,
            'max_depth': 7,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        if params:
            default_params.update(params)
        return lgb.LGBMRegressor(**default_params)
    
    @staticmethod
    def create_catboost(params: Optional[Dict] = None):
        """Create CatBoost regressor with Poisson objective."""
        if not HAS_CATBOOST:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        
        default_params = {
            'loss_function': 'Poisson',
            'iterations': 200,
            'depth': 7,
            'learning_rate': 0.1,
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        }
        if params:
            default_params.update(params)
        return cb.CatBoostRegressor(**default_params)
    
    @staticmethod
    def create_quantile_rf(params: Optional[Dict] = None) -> QuantileRandomForest:
        """Create Quantile Random Forest."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'quantiles': [0.1, 0.5, 0.9]
        }
        if params:
            default_params.update(params)
        return QuantileRandomForest(**default_params)


class ModelSuite:
    """
    Complete model suite for crash risk prediction.
    
    Trains and evaluates multiple models with a unified interface.
    Supports model comparison, ensemble creation, and feature importance.
    """
    
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, ModelResult] = {}
        self.ensemble = None
        
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        include_ensemble: bool = True
    ) -> pd.DataFrame:
        """
        Train all available models and return comparison metrics.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        print("=" * 60)
        print("Training Model Suite")
        print("=" * 60)
        
        results_list = []
        
        # 1. XGBoost
        print("\n[1/5] Training XGBoost...")
        xgb_model = ModelFactory.create_xgboost()
        xgb_model.fit(X_train, y_train)
        xgb_pred = np.clip(xgb_model.predict(X_test), 0, None)
        self.models['xgboost'] = xgb_model
        
        xgb_metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'MAE': mean_absolute_error(y_test, xgb_pred)
        }
        self.results['xgboost'] = ModelResult(
            name='XGBoost',
            model=xgb_model,
            train_predictions=xgb_model.predict(X_train),
            test_predictions=xgb_pred,
            metrics=xgb_metrics,
            feature_importance=self._get_feature_importance(xgb_model, 'xgboost')
        )
        results_list.append({'Model': 'XGBoost', **xgb_metrics})
        print(f"    RMSE: {xgb_metrics['RMSE']:.4f}, MAE: {xgb_metrics['MAE']:.4f}")
        
        # 2. LightGBM
        if HAS_LIGHTGBM:
            print("\n[2/5] Training LightGBM...")
            lgb_model = ModelFactory.create_lightgbm()
            lgb_model.fit(X_train, y_train)
            lgb_pred = np.clip(lgb_model.predict(X_test), 0, None)
            self.models['lightgbm'] = lgb_model
            
            lgb_metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, lgb_pred)),
                'MAE': mean_absolute_error(y_test, lgb_pred)
            }
            self.results['lightgbm'] = ModelResult(
                name='LightGBM',
                model=lgb_model,
                train_predictions=lgb_model.predict(X_train),
                test_predictions=lgb_pred,
                metrics=lgb_metrics,
                feature_importance=self._get_feature_importance(lgb_model, 'lightgbm')
            )
            results_list.append({'Model': 'LightGBM', **lgb_metrics})
            print(f"    RMSE: {lgb_metrics['RMSE']:.4f}, MAE: {lgb_metrics['MAE']:.4f}")
        else:
            print("\n[2/5] Skipping LightGBM (not installed)")
        
        # 3. CatBoost
        if HAS_CATBOOST:
            print("\n[3/5] Training CatBoost...")
            cb_model = ModelFactory.create_catboost()
            cb_model.fit(X_train, y_train)
            cb_pred = np.clip(cb_model.predict(X_test), 0, None)
            self.models['catboost'] = cb_model
            
            cb_metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, cb_pred)),
                'MAE': mean_absolute_error(y_test, cb_pred)
            }
            self.results['catboost'] = ModelResult(
                name='CatBoost',
                model=cb_model,
                train_predictions=cb_model.predict(X_train),
                test_predictions=cb_pred,
                metrics=cb_metrics,
                feature_importance=self._get_feature_importance(cb_model, 'catboost')
            )
            results_list.append({'Model': 'CatBoost', **cb_metrics})
            print(f"    RMSE: {cb_metrics['RMSE']:.4f}, MAE: {cb_metrics['MAE']:.4f}")
        else:
            print("\n[3/5] Skipping CatBoost (not installed)")
        
        # 4. Quantile Random Forest
        print("\n[4/5] Training Quantile Random Forest...")
        qrf_model = ModelFactory.create_quantile_rf()
        qrf_model.fit(X_train, y_train)
        qrf_pred = np.clip(qrf_model.predict(X_test), 0, None)
        self.models['quantile_rf'] = qrf_model
        
        qrf_metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, qrf_pred)),
            'MAE': mean_absolute_error(y_test, qrf_pred)
        }
        self.results['quantile_rf'] = ModelResult(
            name='Quantile RF',
            model=qrf_model,
            train_predictions=qrf_model.predict(X_train),
            test_predictions=qrf_pred,
            metrics=qrf_metrics
        )
        results_list.append({'Model': 'Quantile RF', **qrf_metrics})
        print(f"    RMSE: {qrf_metrics['RMSE']:.4f}, MAE: {qrf_metrics['MAE']:.4f}")
        
        # 5. Stacking Ensemble
        if include_ensemble and len(self.models) >= 2:
            print("\n[5/5] Training Stacking Ensemble...")
            self._train_ensemble(X_train, y_train, X_test, y_test)
            ens_pred = np.clip(self.ensemble.predict(X_test), 0, None)
            
            ens_metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, ens_pred)),
                'MAE': mean_absolute_error(y_test, ens_pred)
            }
            self.results['ensemble'] = ModelResult(
                name='Stacking Ensemble',
                model=self.ensemble,
                train_predictions=self.ensemble.predict(X_train),
                test_predictions=ens_pred,
                metrics=ens_metrics
            )
            results_list.append({'Model': 'Stacking Ensemble', **ens_metrics})
            print(f"    RMSE: {ens_metrics['RMSE']:.4f}, MAE: {ens_metrics['MAE']:.4f}")
        
        comparison_df = pd.DataFrame(results_list)
        comparison_df = comparison_df.sort_values('RMSE').reset_index(drop=True)
        
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def _train_ensemble(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """Train a stacking ensemble from available models."""
        estimators = []
        
        if 'xgboost' in self.models:
            estimators.append(('xgb', ModelFactory.create_xgboost()))
        if 'lightgbm' in self.models:
            estimators.append(('lgb', ModelFactory.create_lightgbm()))
        if 'catboost' in self.models:
            estimators.append(('cb', ModelFactory.create_catboost()))
        
        # Add Random Forest as another base model
        estimators.append(('rf', RandomForestRegressor(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
        )))
        
        self.ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1,
            passthrough=False
        )
        
        self.ensemble.fit(X_train, y_train)
    
    def _get_feature_importance(self, model, model_type: str) -> pd.DataFrame:
        """Extract feature importance from model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
        else:
            return None
        
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Return the best model based on RMSE."""
        best_name = min(self.results, key=lambda x: self.results[x].metrics['RMSE'])
        return best_name, self.models.get(best_name, self.ensemble)
    
    def get_prediction_intervals(
        self, 
        X: pd.DataFrame,
        model_name: str = 'quantile_rf'
    ) -> Dict[float, np.ndarray]:
        """Get prediction intervals from Quantile RF."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        if hasattr(model, 'predict_intervals'):
            return model.predict_intervals(X)
        else:
            raise ValueError(f"Model {model_name} does not support prediction intervals")
    
    def save_models(self, output_dir: str):
        """Save all trained models to directory."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            path = os.path.join(output_dir, f'{name}_model.joblib')
            joblib.dump(model, path)
            print(f"Saved {name} to {path}")
        
        if self.ensemble:
            path = os.path.join(output_dir, 'ensemble_model.joblib')
            joblib.dump(self.ensemble, path)
            print(f"Saved ensemble to {path}")
    
    def load_models(self, model_dir: str):
        """Load models from directory."""
        import os
        
        for filename in os.listdir(model_dir):
            if filename.endswith('_model.joblib'):
                name = filename.replace('_model.joblib', '')
                path = os.path.join(model_dir, filename)
                self.models[name] = joblib.load(path)
                print(f"Loaded {name} from {path}")


def compare_feature_importance(suite: ModelSuite) -> pd.DataFrame:
    """
    Compare feature importance across all models.
    
    Returns a DataFrame with features as rows and models as columns.
    """
    importance_data = {}
    
    for name, result in suite.results.items():
        if result.feature_importance is not None:
            imp = result.feature_importance.set_index('feature')['importance']
            importance_data[result.name] = imp
    
    if not importance_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(importance_data)
    
    # Normalize to percentages
    df = df.div(df.sum(axis=0), axis=1) * 100
    
    # Add average ranking
    df['Avg_Rank'] = df.rank(ascending=False).mean(axis=1)
    df = df.sort_values('Avg_Rank')
    
    return df


if __name__ == '__main__':
    # Example usage
    print("Advanced Models Module")
    print("Available models:", ['XGBoost', 'LightGBM', 'CatBoost', 'Quantile RF', 'Stacking Ensemble'])
    print("LightGBM available:", HAS_LIGHTGBM)
    print("CatBoost available:", HAS_CATBOOST)
