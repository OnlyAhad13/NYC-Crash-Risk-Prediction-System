"""
Anomaly Detection Module for NYC Crash Risk Prediction

Provides anomaly detection for unusual risk patterns:
- Isolation Forest for multivariate anomalies
- Statistical z-score deviation from historical baseline
- Contextual anomalies (unusual for given conditions)
- Alert generation and ranking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AnomalyAlert:
    """Container for an anomaly alert."""
    h3_index: str
    prediction: float
    expected_value: float
    deviation_score: float
    anomaly_type: str
    severity: str
    features: Dict[str, float]
    explanation: str


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection based on historical baselines.
    
    Identifies predictions that deviate significantly from
    expected values based on historical patterns.
    """
    
    def __init__(
        self, 
        threshold_std: float = 2.0,
        min_samples: int = 10
    ):
        """
        Initialize Statistical Anomaly Detector.
        
        Args:
            threshold_std: Number of standard deviations for anomaly
            min_samples: Minimum samples required for baseline
        """
        self.threshold_std = threshold_std
        self.min_samples = min_samples
        self.baselines: Dict[str, Dict] = {}
        
    def fit(
        self, 
        df: pd.DataFrame,
        target_col: str = 'accident_count',
        group_cols: List[str] = ['h3_index', 'hour_of_day']
    ):
        """
        Fit baselines from historical data.
        
        Args:
            df: Historical DataFrame
            target_col: Target column
            group_cols: Columns to group by for baselines
        """
        print(f"Fitting baselines by: {group_cols}")
        
        for group_key, group_df in df.groupby(group_cols):
            if len(group_df) >= self.min_samples:
                baseline = {
                    'mean': group_df[target_col].mean(),
                    'std': max(group_df[target_col].std(), 0.01),  # Avoid zero std
                    'median': group_df[target_col].median(),
                    'p95': group_df[target_col].quantile(0.95),
                    'count': len(group_df)
                }
                self.baselines[str(group_key)] = baseline
        
        print(f"Created {len(self.baselines)} baselines")
        
        return self
    
    def detect(
        self, 
        df: pd.DataFrame,
        prediction_col: str = 'predicted_risk',
        group_cols: List[str] = ['h3_index', 'hour_of_day']
    ) -> pd.DataFrame:
        """
        Detect anomalies in predictions.
        
        Args:
            df: DataFrame with predictions
            prediction_col: Prediction column
            group_cols: Grouping columns matching fit
            
        Returns:
            DataFrame with anomaly scores
        """
        df = df.copy()
        
        z_scores = []
        baselines = []
        is_anomaly = []
        
        for idx, row in df.iterrows():
            group_key = tuple(row[col] for col in group_cols)
            key_str = str(group_key)
            
            if key_str in self.baselines:
                baseline = self.baselines[key_str]
                z = (row[prediction_col] - baseline['mean']) / baseline['std']
                z_scores.append(z)
                baselines.append(baseline['mean'])
                is_anomaly.append(abs(z) > self.threshold_std)
            else:
                z_scores.append(0)
                baselines.append(row[prediction_col])
                is_anomaly.append(False)
        
        df['z_score'] = z_scores
        df['expected_baseline'] = baselines
        df['is_statistical_anomaly'] = is_anomaly
        df['deviation_from_baseline'] = df[prediction_col] - df['expected_baseline']
        
        n_anomalies = df['is_statistical_anomaly'].sum()
        print(f"Detected {n_anomalies} statistical anomalies ({n_anomalies/len(df)*100:.2f}%)")
        
        return df


class IsolationForestDetector:
    """
    Multivariate anomaly detection using Isolation Forest.
    
    Detects anomalies in the feature space, identifying
    unusual combinations of conditions.
    """
    
    def __init__(
        self, 
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest Detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees
            random_state: Random seed
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        
    def fit(
        self, 
        X: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ):
        """
        Fit Isolation Forest model.
        
        Args:
            X: Features DataFrame
            feature_cols: Columns to use (all numeric if None)
        """
        if feature_cols is None:
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_cols = feature_cols
        
        X_scaled = self.scaler.fit_transform(X[feature_cols].fillna(0))
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled)
        
        print(f"Fitted Isolation Forest on {len(feature_cols)} features")
        
        return self
    
    def detect(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in new data.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with anomaly predictions
        """
        X = X.copy()
        
        X_scaled = self.scaler.transform(X[self.feature_cols].fillna(0))
        
        # -1 for anomalies, 1 for normal
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        X['is_isolation_anomaly'] = predictions == -1
        X['isolation_score'] = -scores  # Higher = more anomalous
        
        n_anomalies = X['is_isolation_anomaly'].sum()
        print(f"Detected {n_anomalies} isolation forest anomalies ({n_anomalies/len(X)*100:.2f}%)")
        
        return X


class ContextualAnomalyDetector:
    """
    Contextual anomaly detection considering conditions.
    
    Identifies when predictions are unusual given the
    specific context (weather, time, location history).
    """
    
    def __init__(self, threshold_percentile: float = 95):
        """
        Initialize Contextual Anomaly Detector.
        
        Args:
            threshold_percentile: Percentile threshold for anomaly
        """
        self.threshold_percentile = threshold_percentile
        self.context_models: Dict[str, Dict] = {}
        
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'accident_count',
        context_features: List[str] = ['hour_of_day', 'is_weekend', 'precipitation']
    ):
        """
        Fit context-specific thresholds.
        
        Args:
            df: Historical DataFrame
            target_col: Target column
            context_features: Features defining context
        """
        print(f"Fitting contextual models for: {context_features}")
        
        # Create context bins
        df = df.copy()
        for feat in context_features:
            if df[feat].dtype in ['float64', 'int64'] and df[feat].nunique() > 10:
                df[f'{feat}_bin'] = pd.qcut(
                    df[feat], q=5, labels=False, duplicates='drop'
                )
            else:
                df[f'{feat}_bin'] = df[feat]
        
        bin_cols = [f'{feat}_bin' for feat in context_features]
        
        for context_key, group_df in df.groupby(bin_cols):
            if len(group_df) >= 20:
                threshold = group_df[target_col].quantile(
                    self.threshold_percentile / 100
                )
                self.context_models[str(context_key)] = {
                    'threshold': threshold,
                    'mean': group_df[target_col].mean(),
                    'std': group_df[target_col].std(),
                    'count': len(group_df)
                }
        
        print(f"Created {len(self.context_models)} context models")
        
        return self
    
    def detect(
        self,
        df: pd.DataFrame,
        prediction_col: str = 'predicted_risk',
        context_features: List[str] = ['hour_of_day', 'is_weekend', 'precipitation']
    ) -> pd.DataFrame:
        """
        Detect contextual anomalies.
        
        Args:
            df: DataFrame with predictions
            prediction_col: Prediction column
            context_features: Context features
            
        Returns:
            DataFrame with contextual anomaly flags
        """
        df = df.copy()
        
        # Create context bins
        for feat in context_features:
            if df[feat].dtype in ['float64', 'int64'] and df[feat].nunique() > 10:
                df[f'{feat}_bin'] = pd.qcut(
                    df[feat], q=5, labels=False, duplicates='drop'
                ).fillna(0).astype(int)
            else:
                df[f'{feat}_bin'] = df[feat]
        
        bin_cols = [f'{feat}_bin' for feat in context_features]
        
        is_contextual_anomaly = []
        context_thresholds = []
        
        for idx, row in df.iterrows():
            context_key = tuple(row[col] for col in bin_cols)
            key_str = str(context_key)
            
            if key_str in self.context_models:
                model = self.context_models[key_str]
                is_anomaly = row[prediction_col] > model['threshold']
                threshold = model['threshold']
            else:
                is_anomaly = False
                threshold = row[prediction_col]
            
            is_contextual_anomaly.append(is_anomaly)
            context_thresholds.append(threshold)
        
        df['is_contextual_anomaly'] = is_contextual_anomaly
        df['context_threshold'] = context_thresholds
        df['exceeds_context'] = df[prediction_col] - df['context_threshold']
        
        # Clean up bin columns
        df = df.drop(columns=[c for c in df.columns if c.endswith('_bin')])
        
        n_anomalies = df['is_contextual_anomaly'].sum()
        print(f"Detected {n_anomalies} contextual anomalies ({n_anomalies/len(df)*100:.2f}%)")
        
        return df


class AnomalyAlertGenerator:
    """
    Generate ranked anomaly alerts for dashboard display.
    """
    
    def __init__(
        self,
        stat_detector: StatisticalAnomalyDetector,
        iso_detector: IsolationForestDetector,
        context_detector: ContextualAnomalyDetector
    ):
        self.stat_detector = stat_detector
        self.iso_detector = iso_detector
        self.context_detector = context_detector
        
    def generate_alerts(
        self,
        df: pd.DataFrame,
        prediction_col: str = 'predicted_risk',
        top_n: int = 10
    ) -> List[AnomalyAlert]:
        """
        Generate ranked anomaly alerts.
        
        Args:
            df: DataFrame with predictions
            prediction_col: Prediction column
            top_n: Number of alerts to return
            
        Returns:
            List of AnomalyAlert objects
        """
        # Combine anomaly flags
        df = df.copy()
        
        # Compute overall anomaly score
        anomaly_score = np.zeros(len(df))
        
        if 'z_score' in df.columns:
            anomaly_score += np.abs(df['z_score'].fillna(0))
        if 'isolation_score' in df.columns:
            anomaly_score += df['isolation_score'].fillna(0) * 2
        if 'exceeds_context' in df.columns:
            anomaly_score += np.maximum(df['exceeds_context'].fillna(0), 0) * 3
        
        df['combined_anomaly_score'] = anomaly_score
        
        # Get top anomalies
        is_any_anomaly = (
            df.get('is_statistical_anomaly', False) |
            df.get('is_isolation_anomaly', False) |
            df.get('is_contextual_anomaly', False)
        )
        
        anomaly_df = df[is_any_anomaly].nlargest(top_n, 'combined_anomaly_score')
        
        alerts = []
        for _, row in anomaly_df.iterrows():
            # Determine anomaly type
            types = []
            if row.get('is_statistical_anomaly', False):
                types.append('statistical')
            if row.get('is_isolation_anomaly', False):
                types.append('multivariate')
            if row.get('is_contextual_anomaly', False):
                types.append('contextual')
            
            # Determine severity
            score = row['combined_anomaly_score']
            if score > 5:
                severity = 'critical'
            elif score > 3:
                severity = 'high'
            elif score > 1:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Generate explanation
            explanation = self._generate_explanation(row, types)
            
            alert = AnomalyAlert(
                h3_index=row.get('h3_index', 'unknown'),
                prediction=row[prediction_col],
                expected_value=row.get('expected_baseline', row[prediction_col]),
                deviation_score=score,
                anomaly_type='+'.join(types),
                severity=severity,
                features={
                    'temperature': row.get('temperature', 0),
                    'precipitation': row.get('precipitation', 0),
                    'hour': row.get('hour_of_day', 0)
                },
                explanation=explanation
            )
            
            alerts.append(alert)
        
        return alerts
    
    def _generate_explanation(self, row: pd.Series, types: List[str]) -> str:
        """Generate human-readable explanation for anomaly."""
        parts = []
        
        if 'statistical' in types:
            z = row.get('z_score', 0)
            direction = 'higher' if z > 0 else 'lower'
            parts.append(f"{abs(z):.1f}σ {direction} than historical average")
        
        if 'contextual' in types:
            exceed = row.get('exceeds_context', 0)
            parts.append(f"exceeds context threshold by {exceed:.2f}")
        
        if 'multivariate' in types:
            parts.append("unusual feature combination")
        
        return "; ".join(parts) if parts else "Anomalous pattern detected"


def create_anomaly_pipeline(
    df_historical: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'accident_count'
) -> AnomalyAlertGenerator:
    """
    Create complete anomaly detection pipeline.
    
    Args:
        df_historical: Historical data for fitting
        feature_cols: Feature columns
        target_col: Target column
        
    Returns:
        Fitted AnomalyAlertGenerator
    """
    print("=" * 60)
    print("Creating Anomaly Detection Pipeline")
    print("=" * 60)
    
    # Statistical detector
    print("\n1. Fitting Statistical Detector...")
    stat_detector = StatisticalAnomalyDetector(threshold_std=2.0)
    stat_detector.fit(df_historical, target_col, ['h3_index', 'hour_of_day'])
    
    # Isolation Forest detector
    print("\n2. Fitting Isolation Forest...")
    iso_detector = IsolationForestDetector(contamination=0.05)
    iso_detector.fit(df_historical, feature_cols)
    
    # Contextual detector
    print("\n3. Fitting Contextual Detector...")
    context_detector = ContextualAnomalyDetector(threshold_percentile=95)
    context_detector.fit(
        df_historical, target_col,
        ['hour_of_day', 'is_weekend', 'precipitation']
    )
    
    # Create alert generator
    generator = AnomalyAlertGenerator(stat_detector, iso_detector, context_detector)
    
    print("\nAnomaly pipeline created successfully!")
    
    return generator


if __name__ == '__main__':
    print("Anomaly Detection Module")
    print("Available detectors:")
    print("  - StatisticalAnomalyDetector: Z-score based")
    print("  - IsolationForestDetector: Multivariate")
    print("  - ContextualAnomalyDetector: Context-aware")
    print("  - AnomalyAlertGenerator: Combined alerts")
