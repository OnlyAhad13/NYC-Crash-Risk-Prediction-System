"""
Optuna Hyperparameter Optimization Module for NYC Crash Risk Prediction

Provides Bayesian optimization for:
- XGBoost hyperparameter tuning
- LightGBM hyperparameter tuning
- CatBoost hyperparameter tuning
- Multi-objective optimization (RMSE + MAE)
- Study visualization and best params export
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

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


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimization for gradient boosting models.
    
    Uses Bayesian optimization with Tree-structured Parzen Estimator (TPE)
    for efficient hyperparameter search.
    """
    
    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 3,
        time_series_cv: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Optuna Optimizer.
        
        Args:
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            time_series_cv: Use time-series CV (no shuffle)
            n_jobs: Number of parallel jobs
            random_state: Random seed
            verbose: Print progress
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna not installed. Run: pip install optuna")
        
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.time_series_cv = time_series_cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        self.best_params: Dict[str, Dict] = {}
        self.studies: Dict[str, optuna.Study] = {}
        
    def optimize_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Tuple[Dict, xgb.XGBRegressor]:
        """
        Optimize XGBoost hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Tuple of (best_params, best_model)
        """
        print("\n" + "=" * 50)
        print("Optimizing XGBoost with Optuna")
        print("=" * 50)
        
        def objective(trial):
            params = {
                'objective': 'count:poisson',
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
            }
            
            model = xgb.XGBRegressor(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            else:
                cv = TimeSeriesSplit(n_splits=self.cv_folds) if self.time_series_cv else self.cv_folds
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring='neg_mean_squared_error',
                    n_jobs=self.n_jobs
                )
                rmse = np.sqrt(-scores.mean())
            
            return rmse
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_warmup_steps=10)
        )
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
            n_jobs=1
        )
        
        self.studies['xgboost'] = study
        self.best_params['xgboost'] = study.best_params
        
        # Train final model with best params
        best_params = {
            'objective': 'count:poisson',
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            **study.best_params
        }
        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        print(f"\nBest XGBoost RMSE: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        return study.best_params, best_model
    
    def optimize_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Tuple[Dict, Any]:
        """
        Optimize LightGBM hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Tuple of (best_params, best_model)
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        
        print("\n" + "=" * 50)
        print("Optimizing LightGBM with Optuna")
        print("=" * 50)
        
        def objective(trial):
            params = {
                'objective': 'poisson',
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            else:
                cv = TimeSeriesSplit(n_splits=self.cv_folds) if self.time_series_cv else self.cv_folds
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring='neg_mean_squared_error',
                    n_jobs=self.n_jobs
                )
                rmse = np.sqrt(-scores.mean())
            
            return rmse
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_warmup_steps=10)
        )
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
            n_jobs=1
        )
        
        self.studies['lightgbm'] = study
        self.best_params['lightgbm'] = study.best_params
        
        best_params = {
            'objective': 'poisson',
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': -1,
            **study.best_params
        }
        best_model = lgb.LGBMRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        print(f"\nBest LightGBM RMSE: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        return study.best_params, best_model
    
    def optimize_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Tuple[Dict, Any]:
        """
        Optimize CatBoost hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Tuple of (best_params, best_model)
        """
        if not HAS_CATBOOST:
            raise ImportError("CatBoost not installed")
        
        print("\n" + "=" * 50)
        print("Optimizing CatBoost with Optuna")
        print("=" * 50)
        
        def objective(trial):
            params = {
                'loss_function': 'Poisson',
                'iterations': trial.suggest_int('iterations', 50, 500),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_seed': self.random_state,
                'verbose': False,
                'thread_count': self.n_jobs
            }
            
            model = cb.CatBoostRegressor(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            else:
                cv = TimeSeriesSplit(n_splits=self.cv_folds) if self.time_series_cv else self.cv_folds
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring='neg_mean_squared_error',
                    n_jobs=self.n_jobs
                )
                rmse = np.sqrt(-scores.mean())
            
            return rmse
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_warmup_steps=10)
        )
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
            n_jobs=1
        )
        
        self.studies['catboost'] = study
        self.best_params['catboost'] = study.best_params
        
        best_params = {
            'loss_function': 'Poisson',
            'random_seed': self.random_state,
            'verbose': False,
            'thread_count': self.n_jobs,
            **study.best_params
        }
        best_model = cb.CatBoostRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        print(f"\nBest CatBoost RMSE: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        return study.best_params, best_model
    
    def optimize_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Tuple[Dict, Any]]:
        """
        Optimize all available models.
        
        Returns:
            Dict mapping model name to (best_params, best_model)
        """
        results = {}
        
        # XGBoost
        try:
            xgb_params, xgb_model = self.optimize_xgboost(X_train, y_train, X_val, y_val)
            results['xgboost'] = (xgb_params, xgb_model)
        except Exception as e:
            print(f"XGBoost optimization failed: {e}")
        
        # LightGBM
        if HAS_LIGHTGBM:
            try:
                lgb_params, lgb_model = self.optimize_lightgbm(X_train, y_train, X_val, y_val)
                results['lightgbm'] = (lgb_params, lgb_model)
            except Exception as e:
                print(f"LightGBM optimization failed: {e}")
        
        # CatBoost
        if HAS_CATBOOST:
            try:
                cb_params, cb_model = self.optimize_catboost(X_train, y_train, X_val, y_val)
                results['catboost'] = (cb_params, cb_model)
            except Exception as e:
                print(f"CatBoost optimization failed: {e}")
        
        return results
    
    def get_optimization_history(self, model_name: str) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with trial history
        """
        if model_name not in self.studies:
            return pd.DataFrame()
        
        study = self.studies[model_name]
        
        history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial': trial.number,
                    'value': trial.value,
                    **trial.params
                })
        
        return pd.DataFrame(history)
    
    def plot_optimization_history(
        self,
        model_name: str,
        save_path: Optional[str] = None
    ):
        """
        Plot optimization history.
        
        Args:
            model_name: Name of the model
            save_path: Optional path to save plot
        """
        if model_name not in self.studies:
            print(f"No study found for {model_name}")
            return
        
        import matplotlib.pyplot as plt
        
        history = self.get_optimization_history(model_name)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Optimization history
        axes[0].plot(history['trial'], history['value'], 'b-o', alpha=0.7)
        axes[0].axhline(
            y=history['value'].min(), 
            color='r', linestyle='--', 
            label=f'Best: {history["value"].min():.4f}'
        )
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title(f'{model_name} Optimization History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Best value over time
        best_so_far = history['value'].cummin()
        axes[1].plot(history['trial'], best_so_far, 'g-', linewidth=2)
        axes[1].set_xlabel('Trial')
        axes[1].set_ylabel('Best RMSE So Far')
        axes[1].set_title(f'{model_name} Convergence')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved optimization history to {save_path}")
        
        plt.close()
    
    def plot_param_importances(
        self,
        model_name: str,
        save_path: Optional[str] = None
    ):
        """
        Plot parameter importance from Optuna study.
        
        Args:
            model_name: Name of the model
            save_path: Optional path to save plot
        """
        if model_name not in self.studies:
            print(f"No study found for {model_name}")
            return
        
        try:
            from optuna.visualization import plot_param_importances as optuna_plot
            fig = optuna_plot(self.studies[model_name])
            
            if save_path:
                fig.write_image(save_path)
                print(f"Saved param importances to {save_path}")
            
            return fig
        except Exception as e:
            print(f"Could not plot param importances: {e}")
    
    def export_best_params(self, output_path: str):
        """
        Export best parameters to JSON.
        
        Args:
            output_path: Path to save JSON file
        """
        import json
        
        with open(output_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        print(f"Saved best params to {output_path}")


def run_full_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    n_trials: int = 30
) -> Dict[str, Any]:
    """
    Convenience function to run full optimization pipeline.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Optional validation features
        y_val: Optional validation targets
        n_trials: Number of trials per model
        
    Returns:
        Dict with optimization results
    """
    print("=" * 60)
    print("Running Full Hyperparameter Optimization")
    print("=" * 60)
    
    optimizer = OptunaOptimizer(
        n_trials=n_trials,
        cv_folds=3,
        time_series_cv=True
    )
    
    results = optimizer.optimize_all(X_train, y_train, X_val, y_val)
    
    # Find best model
    best_rmse = float('inf')
    best_name = None
    
    for name, (params, model) in results.items():
        if hasattr(model, 'predict'):
            if X_val is not None:
                pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, pred))
            else:
                rmse = optimizer.studies[name].best_value
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
    
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print(f"Best model: {best_name} (RMSE: {best_rmse:.4f})")
    
    return {
        'optimizer': optimizer,
        'results': results,
        'best_model_name': best_name,
        'best_rmse': best_rmse
    }


if __name__ == '__main__':
    print("Optuna Hyperparameter Optimization Module")
    print("Optuna available:", HAS_OPTUNA)
    print("LightGBM available:", HAS_LIGHTGBM)
    print("CatBoost available:", HAS_CATBOOST)
