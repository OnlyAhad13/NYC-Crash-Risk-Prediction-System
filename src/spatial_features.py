"""
Spatial Features Module for NYC Crash Risk Prediction

Provides advanced spatial features:
- Spatial Lag Features (neighbor hexagon accident counts)
- H3 ring neighbors at multiple distances
- Regional aggregates (borough-level features)
- Spatial autocorrelation metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import h3
import warnings
warnings.filterwarnings('ignore')


class SpatialFeatureGenerator:
    """
    Generate spatial features for H3 hexagonal grid data.
    
    Creates features capturing spatial relationships between
    neighboring hexagons.
    """
    
    def __init__(self):
        self.neighbor_cache: Dict[str, List[str]] = {}
        self.hexagon_stats: Optional[pd.DataFrame] = None
        
    def get_neighbors(
        self, 
        h3_index: str, 
        k: int = 1
    ) -> List[str]:
        """
        Get k-ring neighbors of a hexagon.
        
        Args:
            h3_index: H3 hexagon index
            k: Ring distance (1 = immediate neighbors)
            
        Returns:
            List of neighbor H3 indices
        """
        cache_key = f"{h3_index}_{k}"
        
        if cache_key not in self.neighbor_cache:
            try:
                # Get k-ring (includes center)
                ring = h3.grid_disk(h3_index, k)
                # Remove center hexagon
                neighbors = [h for h in ring if h != h3_index]
                self.neighbor_cache[cache_key] = neighbors
            except Exception:
                self.neighbor_cache[cache_key] = []
        
        return self.neighbor_cache[cache_key]
    
    def compute_hexagon_statistics(
        self, 
        df: pd.DataFrame,
        target_col: str = 'accident_count'
    ) -> pd.DataFrame:
        """
        Compute per-hexagon statistics for use in spatial lag features.
        
        Args:
            df: DataFrame with h3_index and target column
            target_col: Target variable column
            
        Returns:
            DataFrame with hexagon statistics
        """
        stats = df.groupby('h3_index').agg({
            target_col: ['mean', 'std', 'sum', 'count']
        }).reset_index()
        
        stats.columns = ['h3_index', 'hex_mean', 'hex_std', 'hex_total', 'hex_count']
        stats['hex_std'] = stats['hex_std'].fillna(0)
        
        self.hexagon_stats = stats
        return stats
    
    def add_spatial_lag_features(
        self, 
        df: pd.DataFrame,
        target_col: str = 'accident_count',
        k_rings: List[int] = [1, 2],
        agg_funcs: List[str] = ['mean', 'sum', 'max']
    ) -> pd.DataFrame:
        """
        Add spatial lag features based on neighbor statistics.
        
        Features include aggregations of neighbor hexagon values.
        
        Args:
            df: Input DataFrame with h3_index
            target_col: Target variable column
            k_rings: List of ring distances to consider
            agg_funcs: Aggregation functions to apply
            
        Returns:
            DataFrame with spatial lag features added
        """
        print(f"Adding spatial lag features for k-rings: {k_rings}")
        
        # Compute hexagon statistics if not already done
        if self.hexagon_stats is None:
            self.compute_hexagon_statistics(df, target_col)
        
        hex_mean_dict = self.hexagon_stats.set_index('h3_index')['hex_mean'].to_dict()
        
        df = df.copy()
        
        for k in k_rings:
            print(f"  Processing k={k} ring...")
            
            neighbor_means = []
            neighbor_sums = []
            neighbor_maxes = []
            neighbor_counts = []
            
            for h3_idx in df['h3_index']:
                neighbors = self.get_neighbors(h3_idx, k)
                
                if neighbors:
                    neighbor_values = [
                        hex_mean_dict.get(n, 0) for n in neighbors
                    ]
                    
                    neighbor_means.append(np.mean(neighbor_values))
                    neighbor_sums.append(np.sum(neighbor_values))
                    neighbor_maxes.append(np.max(neighbor_values))
                    neighbor_counts.append(len(neighbors))
                else:
                    neighbor_means.append(0)
                    neighbor_sums.append(0)
                    neighbor_maxes.append(0)
                    neighbor_counts.append(0)
            
            if 'mean' in agg_funcs:
                df[f'neighbor_mean_k{k}'] = neighbor_means
            if 'sum' in agg_funcs:
                df[f'neighbor_sum_k{k}'] = neighbor_sums
            if 'max' in agg_funcs:
                df[f'neighbor_max_k{k}'] = neighbor_maxes
            
            df[f'neighbor_count_k{k}'] = neighbor_counts
        
        return df
    
    def add_distance_to_hotspot(
        self, 
        df: pd.DataFrame,
        n_hotspots: int = 10
    ) -> pd.DataFrame:
        """
        Add feature for distance to nearest hotspot.
        
        Args:
            df: DataFrame with h3_index
            n_hotspots: Number of top hotspots to consider
            
        Returns:
            DataFrame with hotspot distance features
        """
        if self.hexagon_stats is None:
            raise ValueError("Call compute_hexagon_statistics first")
        
        # Identify top hotspots
        hotspots = self.hexagon_stats.nlargest(n_hotspots, 'hex_mean')['h3_index'].tolist()
        
        df = df.copy()
        
        min_distances = []
        for h3_idx in df['h3_index']:
            if h3_idx in hotspots:
                min_distances.append(0)
            else:
                # Calculate grid distance to nearest hotspot
                distances = []
                for hotspot in hotspots:
                    try:
                        dist = h3.grid_distance(h3_idx, hotspot)
                        distances.append(dist)
                    except Exception:
                        distances.append(100)  # Large distance if error
                min_distances.append(min(distances) if distances else 100)
        
        df['dist_to_hotspot'] = min_distances
        df['is_hotspot'] = (df['dist_to_hotspot'] == 0).astype(int)
        
        return df


class RegionalAggregator:
    """
    Create regional aggregate features for hexagons.
    
    Groups hexagons into larger regions and computes
    regional statistics.
    """
    
    def __init__(self):
        self.region_mapping: Dict[str, str] = {}
        self.region_stats: Optional[pd.DataFrame] = None
        
    def assign_regions_by_parent(
        self, 
        df: pd.DataFrame,
        parent_resolution: int = 5
    ) -> pd.DataFrame:
        """
        Assign hexagons to regions based on parent hexagon.
        
        Args:
            df: DataFrame with h3_index
            parent_resolution: H3 resolution for parent regions
            
        Returns:
            DataFrame with region_id column
        """
        df = df.copy()
        
        regions = []
        for h3_idx in df['h3_index']:
            try:
                parent = h3.cell_to_parent(h3_idx, parent_resolution)
                regions.append(parent)
                self.region_mapping[h3_idx] = parent
            except Exception:
                regions.append('unknown')
        
        df['region_id'] = regions
        
        n_regions = df['region_id'].nunique()
        print(f"Assigned {len(df)} hexagons to {n_regions} regions")
        
        return df
    
    def compute_region_features(
        self, 
        df: pd.DataFrame,
        target_col: str = 'accident_count'
    ) -> pd.DataFrame:
        """
        Compute region-level aggregate features.
        
        Args:
            df: DataFrame with region_id and target
            target_col: Target column
            
        Returns:
            DataFrame with region features added
        """
        if 'region_id' not in df.columns:
            raise ValueError("Call assign_regions_by_parent first")
        
        # Compute region statistics
        region_stats = df.groupby('region_id').agg({
            target_col: ['mean', 'std', 'sum'],
            'h3_index': 'nunique'
        }).reset_index()
        
        region_stats.columns = [
            'region_id', 'region_mean', 'region_std', 
            'region_total', 'region_hex_count'
        ]
        region_stats['region_std'] = region_stats['region_std'].fillna(0)
        
        self.region_stats = region_stats
        
        # Merge back to main DataFrame
        df = df.merge(region_stats, on='region_id', how='left')
        
        # Add relative features
        df['rel_to_region'] = df[target_col] / df['region_mean'].clip(lower=0.001)
        
        return df


class SpatialAutocorrelation:
    """
    Compute spatial autocorrelation metrics.
    
    Measures how similar nearby hexagons are in their values.
    """
    
    @staticmethod
    def compute_morans_i(
        df: pd.DataFrame,
        value_col: str = 'accident_count',
        h3_col: str = 'h3_index',
        k: int = 1
    ) -> Dict[str, float]:
        """
        Compute Moran's I spatial autocorrelation.
        
        Args:
            df: DataFrame with hexagon values
            value_col: Column with values to test
            h3_col: H3 index column
            k: Neighbor ring distance
            
        Returns:
            Dict with Moran's I statistics
        """
        # Get unique hexagons and their mean values
        hex_values = df.groupby(h3_col)[value_col].mean()
        n = len(hex_values)
        
        if n < 10:
            return {'morans_i': 0, 'expected_i': 0, 'interpretation': 'insufficient_data'}
        
        mean_val = hex_values.mean()
        
        # Compute spatial weights and cross-product
        numerator = 0
        denominator = 0
        w_sum = 0
        
        for h3_idx, value in hex_values.items():
            deviation = value - mean_val
            denominator += deviation ** 2
            
            # Get neighbors
            try:
                neighbors = h3.grid_disk(h3_idx, k)
                neighbors = [n for n in neighbors if n != h3_idx and n in hex_values.index]
            except Exception:
                neighbors = []
            
            for neighbor in neighbors:
                neighbor_deviation = hex_values[neighbor] - mean_val
                numerator += deviation * neighbor_deviation
                w_sum += 1
        
        if denominator == 0 or w_sum == 0:
            return {'morans_i': 0, 'expected_i': -1/(n-1), 'interpretation': 'no_variance'}
        
        morans_i = (n / w_sum) * (numerator / denominator)
        expected_i = -1 / (n - 1)
        
        # Interpretation
        if morans_i > 0.3:
            interpretation = 'strong_positive_clustering'
        elif morans_i > 0.1:
            interpretation = 'moderate_positive_clustering'
        elif morans_i > -0.1:
            interpretation = 'random_pattern'
        else:
            interpretation = 'dispersed_pattern'
        
        return {
            'morans_i': morans_i,
            'expected_i': expected_i,
            'interpretation': interpretation,
            'n_hexagons': n,
            'n_connections': w_sum
        }
    
    @staticmethod
    def compute_local_morans(
        df: pd.DataFrame,
        value_col: str = 'accident_count',
        h3_col: str = 'h3_index',
        k: int = 1
    ) -> pd.DataFrame:
        """
        Compute Local Moran's I for each hexagon.
        
        Identifies local clusters and outliers.
        
        Args:
            df: DataFrame with hexagon values
            value_col: Value column
            h3_col: H3 index column
            k: Neighbor distance
            
        Returns:
            DataFrame with local Moran's I values
        """
        # Aggregate to hexagon level
        hex_df = df.groupby(h3_col)[value_col].mean().reset_index()
        hex_df.columns = [h3_col, 'value']
        
        mean_val = hex_df['value'].mean()
        std_val = hex_df['value'].std()
        
        if std_val == 0:
            hex_df['local_moran'] = 0
            hex_df['cluster_type'] = 'no_variance'
            return hex_df
        
        hex_df['z_score'] = (hex_df['value'] - mean_val) / std_val
        
        value_dict = hex_df.set_index(h3_col)['z_score'].to_dict()
        
        local_morans = []
        cluster_types = []
        
        for _, row in hex_df.iterrows():
            h3_idx = row[h3_col]
            z_i = row['z_score']
            
            try:
                neighbors = h3.grid_disk(h3_idx, k)
                neighbors = [n for n in neighbors if n != h3_idx and n in value_dict]
            except Exception:
                neighbors = []
            
            if neighbors:
                neighbor_z = np.mean([value_dict[n] for n in neighbors])
                local_i = z_i * neighbor_z
            else:
                local_i = 0
                neighbor_z = 0
            
            local_morans.append(local_i)
            
            # Classify cluster type
            if local_i > 0.5 and z_i > 0:
                cluster_types.append('high_high')
            elif local_i > 0.5 and z_i < 0:
                cluster_types.append('low_low')
            elif local_i < -0.5 and z_i > 0:
                cluster_types.append('high_low_outlier')
            elif local_i < -0.5 and z_i < 0:
                cluster_types.append('low_high_outlier')
            else:
                cluster_types.append('not_significant')
        
        hex_df['local_moran'] = local_morans
        hex_df['cluster_type'] = cluster_types
        
        return hex_df


def add_all_spatial_features(
    df: pd.DataFrame,
    target_col: str = 'accident_count',
    k_rings: List[int] = [1, 2]
) -> pd.DataFrame:
    """
    Convenience function to add all spatial features.
    
    Args:
        df: Input DataFrame
        target_col: Target column
        k_rings: Ring distances for neighbor features
        
    Returns:
        DataFrame with all spatial features
    """
    print("=" * 60)
    print("Adding Spatial Features")
    print("=" * 60)
    
    # 1. Spatial lag features
    sfg = SpatialFeatureGenerator()
    sfg.compute_hexagon_statistics(df, target_col)
    df = sfg.add_spatial_lag_features(df, target_col, k_rings)
    
    # 2. Hotspot distance
    print("\nAdding hotspot distance features...")
    df = sfg.add_distance_to_hotspot(df, n_hotspots=10)
    
    # 3. Regional features
    print("\nAdding regional aggregate features...")
    ra = RegionalAggregator()
    df = ra.assign_regions_by_parent(df, parent_resolution=5)
    df = ra.compute_region_features(df, target_col)
    
    # Summary
    spatial_cols = [c for c in df.columns if 'neighbor' in c or 'region' in c or 'hotspot' in c or 'dist_to' in c]
    print(f"\nAdded {len(spatial_cols)} spatial features:")
    for col in spatial_cols:
        print(f"  - {col}")
    
    return df


if __name__ == '__main__':
    print("Spatial Features Module")
    print("Available classes:")
    print("  - SpatialFeatureGenerator: Neighbor lag features")
    print("  - RegionalAggregator: Regional statistics")
    print("  - SpatialAutocorrelation: Moran's I analysis")
