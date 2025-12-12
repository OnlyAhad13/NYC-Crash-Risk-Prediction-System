"""
Clustering Module for NYC Crash Risk Prediction

Provides hotspot pattern analysis:
- K-Means clustering on hexagon features
- DBSCAN for spatial density clustering
- Cluster profiling and interpretation
- Temporal pattern extraction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class HexagonClusterer:
    """
    Cluster hexagons based on their risk profiles.
    
    Groups similar hexagons to identify patterns and
    enable targeted interventions.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.feature_cols: List[str] = []
        self.cluster_profiles: Optional[pd.DataFrame] = None
        
    def prepare_hexagon_features(
        self, 
        df: pd.DataFrame,
        h3_col: str = 'h3_index',
        target_col: str = 'accident_count',
        agg_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate data to hexagon level for clustering.
        
        Args:
            df: Input DataFrame
            h3_col: H3 index column
            target_col: Target column
            agg_features: Features to aggregate
            
        Returns:
            Hexagon-level feature DataFrame
        """
        if agg_features is None:
            agg_features = [
                'temperature', 'precipitation', 'wind_speed',
                'hour_of_day', 'is_weekend', 'is_holiday'
            ]
        
        # Define aggregations
        agg_dict = {target_col: ['mean', 'std', 'max', 'sum', 'count']}
        
        for feat in agg_features:
            if feat in df.columns:
                agg_dict[feat] = ['mean']
        
        # Aggregate
        hex_df = df.groupby(h3_col).agg(agg_dict).reset_index()
        
        # Flatten column names
        hex_df.columns = [
            f'{col[0]}_{col[1]}' if col[1] else col[0] 
            for col in hex_df.columns
        ]
        hex_df = hex_df.rename(columns={f'{h3_col}_': h3_col})
        
        # Add derived features
        hex_df['risk_variance'] = hex_df[f'{target_col}_std'] / (hex_df[f'{target_col}_mean'] + 0.01)
        hex_df['peak_ratio'] = hex_df[f'{target_col}_max'] / (hex_df[f'{target_col}_mean'] + 0.01)
        
        # Fill NaN
        hex_df = hex_df.fillna(0)
        
        self.feature_cols = [c for c in hex_df.columns if c != h3_col and c != f'{target_col}_count']
        
        print(f"Prepared {len(hex_df)} hexagons with {len(self.feature_cols)} features")
        
        return hex_df
    
    def fit_kmeans(
        self, 
        hex_df: pd.DataFrame,
        n_clusters: int = 5,
        h3_col: str = 'h3_index'
    ) -> pd.DataFrame:
        """
        Fit K-Means clustering.
        
        Args:
            hex_df: Hexagon-level features
            n_clusters: Number of clusters
            h3_col: H3 index column
            
        Returns:
            DataFrame with cluster assignments
        """
        X = hex_df[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        clusters = self.cluster_model.fit_predict(X_scaled)
        
        hex_df = hex_df.copy()
        hex_df['cluster'] = clusters
        
        # Compute cluster profiles
        self._compute_cluster_profiles(hex_df, h3_col)
        
        print(f"\nK-Means clustering complete:")
        for i in range(n_clusters):
            count = (clusters == i).sum()
            print(f"  Cluster {i}: {count} hexagons ({count/len(clusters)*100:.1f}%)")
        
        return hex_df
    
    def fit_dbscan(
        self, 
        hex_df: pd.DataFrame,
        eps: float = 0.5,
        min_samples: int = 5,
        h3_col: str = 'h3_index'
    ) -> pd.DataFrame:
        """
        Fit DBSCAN density clustering.
        
        Args:
            hex_df: Hexagon-level features
            eps: Maximum distance between samples
            min_samples: Minimum cluster size
            h3_col: H3 index column
            
        Returns:
            DataFrame with cluster assignments
        """
        X = hex_df[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.cluster_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1
        )
        
        clusters = self.cluster_model.fit_predict(X_scaled)
        
        hex_df = hex_df.copy()
        hex_df['cluster'] = clusters
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = (clusters == -1).sum()
        
        print(f"\nDBSCAN clustering complete:")
        print(f"  Found {n_clusters} clusters")
        print(f"  Noise points: {n_noise} ({n_noise/len(clusters)*100:.1f}%)")
        
        return hex_df
    
    def _compute_cluster_profiles(
        self, 
        hex_df: pd.DataFrame,
        h3_col: str = 'h3_index'
    ):
        """Compute cluster profiles for interpretation."""
        
        profiles = []
        
        for cluster_id in sorted(hex_df['cluster'].unique()):
            cluster_data = hex_df[hex_df['cluster'] == cluster_id]
            
            profile = {
                'cluster': cluster_id,
                'size': len(cluster_data),
                'pct_of_total': len(cluster_data) / len(hex_df) * 100
            }
            
            # Add feature means
            for col in self.feature_cols:
                profile[f'{col}_mean'] = cluster_data[col].mean()
                profile[f'{col}_std'] = cluster_data[col].std()
            
            profiles.append(profile)
        
        self.cluster_profiles = pd.DataFrame(profiles)
    
    def get_cluster_interpretation(self) -> Dict[int, str]:
        """
        Generate human-readable cluster interpretations.
        
        Returns:
            Dict mapping cluster ID to description
        """
        if self.cluster_profiles is None:
            return {}
        
        interpretations = {}
        
        for _, row in self.cluster_profiles.iterrows():
            cluster_id = int(row['cluster'])
            
            # Find distinguishing features
            risk_mean = row.get('accident_count_mean_mean', 0)
            risk_var = row.get('risk_variance_mean', 0)
            
            # Generate description
            if risk_mean > self.cluster_profiles['accident_count_mean_mean'].quantile(0.75):
                risk_level = "High-risk"
            elif risk_mean < self.cluster_profiles['accident_count_mean_mean'].quantile(0.25):
                risk_level = "Low-risk"
            else:
                risk_level = "Medium-risk"
            
            if risk_var > 1.5:
                pattern = "volatile"
            elif risk_var < 0.5:
                pattern = "stable"
            else:
                pattern = "moderate variability"
            
            interpretations[cluster_id] = f"{risk_level} areas with {pattern} patterns"
        
        return interpretations
    
    def find_optimal_k(
        self, 
        hex_df: pd.DataFrame,
        k_range: range = range(2, 11),
        method: str = 'elbow'
    ) -> Tuple[int, pd.DataFrame]:
        """
        Find optimal number of clusters.
        
        Args:
            hex_df: Hexagon-level features
            k_range: Range of k values to try
            method: 'elbow' or 'silhouette'
            
        Returns:
            Tuple of (optimal_k, evaluation_df)
        """
        from sklearn.metrics import silhouette_score
        
        X = hex_df[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        results = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            inertia = kmeans.inertia_
            
            if k > 1:
                silhouette = silhouette_score(X_scaled, labels)
            else:
                silhouette = 0
            
            results.append({
                'k': k,
                'inertia': inertia,
                'silhouette': silhouette
            })
        
        eval_df = pd.DataFrame(results)
        
        # Find optimal k
        if method == 'silhouette':
            optimal_k = eval_df.loc[eval_df['silhouette'].idxmax(), 'k']
        else:  # elbow method
            # Find elbow using simple heuristic
            eval_df['inertia_diff'] = eval_df['inertia'].diff()
            eval_df['inertia_diff2'] = eval_df['inertia_diff'].diff()
            
            # Where second derivative is smallest (most negative)
            if len(eval_df) > 2:
                optimal_k = eval_df.iloc[2:].loc[
                    eval_df.iloc[2:]['inertia_diff2'].abs().idxmin(), 'k'
                ]
            else:
                optimal_k = k_range[len(k_range)//2]
        
        print(f"Optimal k by {method}: {optimal_k}")
        
        return int(optimal_k), eval_df


class TemporalPatternExtractor:
    """
    Extract temporal patterns from hexagon data.
    
    Identifies when each hexagon has peak risk and
    characterizes temporal profiles.
    """
    
    def __init__(self):
        self.hourly_profiles: Optional[pd.DataFrame] = None
        self.daily_profiles: Optional[pd.DataFrame] = None
        
    def extract_hourly_patterns(
        self, 
        df: pd.DataFrame,
        h3_col: str = 'h3_index',
        target_col: str = 'accident_count'
    ) -> pd.DataFrame:
        """
        Extract hourly risk patterns per hexagon.
        
        Args:
            df: Input DataFrame with hour_of_day column
            h3_col: H3 index column
            target_col: Target column
            
        Returns:
            DataFrame with hourly patterns
        """
        if 'hour_of_day' not in df.columns:
            raise ValueError("DataFrame must have 'hour_of_day' column")
        
        # Pivot to get hourly profiles
        hourly = df.groupby([h3_col, 'hour_of_day'])[target_col].mean().reset_index()
        hourly_pivot = hourly.pivot(
            index=h3_col, 
            columns='hour_of_day', 
            values=target_col
        ).fillna(0)
        
        # Add derived features
        hourly_pivot.columns = [f'hour_{h}' for h in hourly_pivot.columns]
        
        hourly_pivot['peak_hour'] = hourly_pivot.idxmax(axis=1).str.replace('hour_', '').astype(int)
        hourly_pivot['trough_hour'] = hourly_pivot.idxmin(axis=1).str.replace('hour_', '').astype(int)
        hourly_pivot['peak_trough_ratio'] = (
            hourly_pivot.max(axis=1) / (hourly_pivot.min(axis=1) + 0.001)
        )
        
        # Morning (6-10), Day (10-16), Evening (16-20), Night (20-6)
        morning_cols = [f'hour_{h}' for h in range(6, 10) if f'hour_{h}' in hourly_pivot.columns]
        day_cols = [f'hour_{h}' for h in range(10, 16) if f'hour_{h}' in hourly_pivot.columns]
        evening_cols = [f'hour_{h}' for h in range(16, 20) if f'hour_{h}' in hourly_pivot.columns]
        night_cols = [f'hour_{h}' for h in list(range(20, 24)) + list(range(0, 6)) if f'hour_{h}' in hourly_pivot.columns]
        
        hourly_pivot['morning_avg'] = hourly_pivot[morning_cols].mean(axis=1) if morning_cols else 0
        hourly_pivot['day_avg'] = hourly_pivot[day_cols].mean(axis=1) if day_cols else 0
        hourly_pivot['evening_avg'] = hourly_pivot[evening_cols].mean(axis=1) if evening_cols else 0
        hourly_pivot['night_avg'] = hourly_pivot[night_cols].mean(axis=1) if night_cols else 0
        
        # Classify pattern type
        def classify_pattern(row):
            period_avgs = {
                'morning': row.get('morning_avg', 0),
                'day': row.get('day_avg', 0),
                'evening': row.get('evening_avg', 0),
                'night': row.get('night_avg', 0)
            }
            max_period = max(period_avgs, key=period_avgs.get)
            return f'{max_period}_peak'
        
        hourly_pivot['temporal_pattern'] = hourly_pivot.apply(classify_pattern, axis=1)
        
        self.hourly_profiles = hourly_pivot.reset_index()
        
        print(f"Extracted hourly patterns for {len(hourly_pivot)} hexagons")
        print("\nTemporal pattern distribution:")
        print(hourly_pivot['temporal_pattern'].value_counts())
        
        return self.hourly_profiles
    
    def extract_weekly_patterns(
        self, 
        df: pd.DataFrame,
        h3_col: str = 'h3_index',
        target_col: str = 'accident_count'
    ) -> pd.DataFrame:
        """
        Extract day-of-week patterns.
        
        Args:
            df: Input DataFrame with day_of_week column
            h3_col: H3 index column
            target_col: Target column
            
        Returns:
            DataFrame with weekly patterns
        """
        if 'day_of_week' not in df.columns:
            raise ValueError("DataFrame must have 'day_of_week' column")
        
        weekly = df.groupby([h3_col, 'day_of_week'])[target_col].mean().reset_index()
        weekly_pivot = weekly.pivot(
            index=h3_col,
            columns='day_of_week',
            values=target_col
        ).fillna(0)
        
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_pivot.columns = [day_names[i] for i in weekly_pivot.columns]
        
        weekly_pivot['weekday_avg'] = weekly_pivot[['Mon', 'Tue', 'Wed', 'Thu', 'Fri']].mean(axis=1)
        weekly_pivot['weekend_avg'] = weekly_pivot[['Sat', 'Sun']].mean(axis=1)
        weekly_pivot['weekend_weekday_ratio'] = (
            weekly_pivot['weekend_avg'] / (weekly_pivot['weekday_avg'] + 0.001)
        )
        
        # Classify
        def classify_weekly(row):
            if row['weekend_weekday_ratio'] > 1.5:
                return 'weekend_dominant'
            elif row['weekend_weekday_ratio'] < 0.67:
                return 'weekday_dominant'
            else:
                return 'balanced'
        
        weekly_pivot['weekly_pattern'] = weekly_pivot.apply(classify_weekly, axis=1)
        
        self.daily_profiles = weekly_pivot.reset_index()
        
        print(f"\nExtracted weekly patterns for {len(weekly_pivot)} hexagons")
        print("\nWeekly pattern distribution:")
        print(weekly_pivot['weekly_pattern'].value_counts())
        
        return self.daily_profiles


class ClusterVisualizer:
    """
    Visualization utilities for clustering results.
    """
    
    @staticmethod
    def plot_cluster_profiles(
        cluster_profiles: pd.DataFrame,
        feature_cols: List[str],
        save_path: Optional[str] = None
    ):
        """
        Create radar/spider chart of cluster profiles.
        
        Args:
            cluster_profiles: Cluster profile DataFrame
            feature_cols: Features to plot
            save_path: Optional path to save
        """
        from math import pi
        
        # Get mean columns
        mean_cols = [f'{c}_mean' for c in feature_cols if f'{c}_mean' in cluster_profiles.columns][:8]
        
        if not mean_cols:
            print("No mean columns found for plotting")
            return
        
        # Normalize for plotting
        plot_df = cluster_profiles[['cluster'] + mean_cols].copy()
        for col in mean_cols:
            col_min = plot_df[col].min()
            col_max = plot_df[col].max()
            if col_max > col_min:
                plot_df[col] = (plot_df[col] - col_min) / (col_max - col_min)
        
        # Create radar chart
        n_clusters = len(plot_df)
        n_vars = len(mean_cols)
        
        angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
        
        for idx, row in plot_df.iterrows():
            values = row[mean_cols].tolist()
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=f'Cluster {int(row["cluster"])}', color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        labels = [c.replace('_mean', '').replace('_', ' ') for c in mean_cols]
        ax.set_xticklabels(labels, size=10)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Cluster Profiles', size=14, y=1.08)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved cluster profiles to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_cluster_pca(
        hex_df: pd.DataFrame,
        feature_cols: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot clusters in PCA space.
        
        Args:
            hex_df: Hexagon data with cluster column
            feature_cols: Feature columns used for clustering
            save_path: Optional path to save
        """
        if 'cluster' not in hex_df.columns:
            print("No cluster column found")
            return
        
        X = hex_df[feature_cols].fillna(0).values
        X_scaled = StandardScaler().fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        
        clusters = hex_df['cluster'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(clusters)))
        
        for i, cluster_id in enumerate(sorted(clusters)):
            mask = hex_df['cluster'] == cluster_id
            plt.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=[colors[i]], label=f'Cluster {cluster_id}',
                alpha=0.6, s=50
            )
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('Hexagon Clusters in PCA Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved PCA plot to {save_path}")
        
        plt.close()


def cluster_hexagons(
    df: pd.DataFrame,
    h3_col: str = 'h3_index',
    target_col: str = 'accident_count',
    n_clusters: int = 5
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Convenience function for complete clustering analysis.
    
    Args:
        df: Input DataFrame
        h3_col: H3 index column
        target_col: Target column
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (clustered_hex_df, cluster_interpretations)
    """
    print("=" * 60)
    print("Hexagon Clustering Analysis")
    print("=" * 60)
    
    clusterer = HexagonClusterer()
    
    # Prepare features
    hex_df = clusterer.prepare_hexagon_features(df, h3_col, target_col)
    
    # Fit K-Means
    hex_df = clusterer.fit_kmeans(hex_df, n_clusters)
    
    # Get interpretations
    interpretations = clusterer.get_cluster_interpretation()
    
    print("\nCluster Interpretations:")
    for cluster_id, desc in interpretations.items():
        print(f"  Cluster {cluster_id}: {desc}")
    
    return hex_df, interpretations


if __name__ == '__main__':
    print("Clustering Module")
    print("Available classes:")
    print("  - HexagonClusterer: K-Means and DBSCAN clustering")
    print("  - TemporalPatternExtractor: Hourly and weekly patterns")
    print("  - ClusterVisualizer: Visualization utilities")
