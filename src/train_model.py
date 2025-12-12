"""
Model Training Script for NYC Crash Risk Prediction

Trains and evaluates:
1. Baseline (mean prediction)
2. Random Forest Regressor
3. XGBoost with hyperparameter tuning

Outputs comparison table, feature importance plot, and saves final model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Sample size for training (to fit in memory)
TRAIN_SAMPLE = 500_000
TEST_SAMPLE = 100_000


def load_data(train_path: str, test_path: str):
    """Load train and test data with sampling for memory efficiency."""
    print("\nLoading data...")
    
    # Load train data (sample if too large)
    train_df = pd.read_csv(train_path, parse_dates=['hour'])
    if len(train_df) > TRAIN_SAMPLE:
        train_df = train_df.sample(n=TRAIN_SAMPLE, random_state=42)
    print(f"  Train samples: {len(train_df):,}")
    
    # Load test data (sample if too large)
    test_df = pd.read_csv(test_path, parse_dates=['hour'])
    if len(test_df) > TEST_SAMPLE:
        test_df = test_df.sample(n=TEST_SAMPLE, random_state=42)
    print(f"  Test samples: {len(test_df):,}")
    
    return train_df, test_df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Prepare feature matrix and target variable."""
    print("\nPreparing features...")
    
    # Feature columns (exclude non-numeric and target)
    feature_cols = [
        'temperature', 'precipitation', 'wind_speed', 'snow_depth',
        'hour_of_day', 'day_of_week', 'month', 'year',
        'accidents_1h_ago', 'accidents_24h_ago', 'rolling_mean_7d'
    ]
    
    # Add boolean features as integers
    for col in ['is_holiday', 'is_weekend']:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(int)
            test_df[col] = test_df[col].astype(int)
            feature_cols.append(col)
    
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    print(f"  Features: {feature_cols}")
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['accident_count']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['accident_count']
    
    return X_train, y_train, X_test, y_test, feature_cols


def train_baseline(y_train, y_test):
    """Calculate baseline metrics using mean prediction."""
    print("\n" + "="*50)
    print("Training Baseline Model (Mean Prediction)")
    print("="*50)
    
    mean_pred = y_train.mean()
    print(f"  Training mean: {mean_pred:.4f}")
    
    # Predict mean for all test samples
    y_pred = np.full(len(y_test), mean_pred)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return {'model': 'Baseline (Mean)', 'RMSE': rmse, 'MAE': mae, 'MSE': mse}


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest Regressor."""
    print("\n" + "="*50)
    print("Training Random Forest Regressor")
    print("="*50)
    
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("  Fitting model...")
    rf.fit(X_train, y_train)
    
    print("  Predicting...")
    y_pred = rf.predict(X_test)
    
    # Clip predictions to non-negative (count data)
    y_pred = np.clip(y_pred, 0, None)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return rf, {'model': 'Random Forest', 'RMSE': rmse, 'MAE': mae, 'MSE': mse}


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with hyperparameter tuning."""
    print("\n" + "="*50)
    print("Training XGBoost with Hyperparameter Tuning")
    print("="*50)
    
    # Define parameter grid for RandomizedSearchCV
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(
        objective='count:poisson',  # Good for count data
        random_state=42,
        n_jobs=-1
    )
    
    print("  Running RandomizedSearchCV (10 iterations, 3-fold CV)...")
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    print(f"\n  Best parameters: {random_search.best_params_}")
    
    print("  Predicting on test set...")
    y_pred = best_model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return best_model, {'model': 'XGBoost (Tuned)', 'RMSE': rmse, 'MAE': mae, 'MSE': mse}


def plot_feature_importance(model, feature_cols, output_path):
    """Plot and save feature importance graph."""
    print("\nPlotting feature importance...")
    
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('XGBoost Feature Importance for Accident Count Prediction')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to: {output_path}")
    
    # Print top features
    print("\n  Top 5 features:")
    for _, row in importance_df.tail(5).iloc[::-1].iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")


def create_comparison_table(results):
    """Create and display comparison table."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    df = pd.DataFrame(results)
    df = df[['model', 'RMSE', 'MAE', 'MSE']]
    
    # Format numbers
    for col in ['RMSE', 'MAE', 'MSE']:
        df[col] = df[col].round(4)
    
    print(df.to_string(index=False))
    
    # Find best model
    best_idx = df['RMSE'].idxmin()
    print(f"\nBest model by RMSE: {df.loc[best_idx, 'model']}")
    
    return df


def save_model(model, output_path):
    """Save trained model to file."""
    print(f"\nSaving model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print("  Model saved successfully!")


def main():
    print("="*60)
    print("NYC Crash Risk Prediction - Model Training")
    print("="*60)
    
    # Paths
    train_path = 'data/processed/train.csv'
    test_path = 'data/processed/test.csv'
    model_path = 'models/final_model.joblib'
    importance_path = 'models/feature_importance.png'
    
    # Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # Prepare features
    X_train, y_train, X_test, y_test, feature_cols = prepare_features(train_df, test_df)
    
    # Store results
    results = []
    
    # 1. Baseline
    baseline_results = train_baseline(y_train, y_test)
    results.append(baseline_results)
    
    # 2. Random Forest
    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    results.append(rf_results)
    
    # 3. XGBoost with tuning
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    results.append(xgb_results)
    
    # Comparison table
    comparison_df = create_comparison_table(results)
    
    # Feature importance plot
    os.makedirs('models', exist_ok=True)
    plot_feature_importance(xgb_model, feature_cols, importance_path)
    
    # Save final XGBoost model
    save_model(xgb_model, model_path)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - Model: {model_path}")
    print(f"  - Feature importance: {importance_path}")


if __name__ == '__main__':
    main()
