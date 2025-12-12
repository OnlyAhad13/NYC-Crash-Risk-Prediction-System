"""
Feature Engineering - Fully Vectorized Version

Uses vectorized pandas operations only. No iterrows.
Computes approximate lag features within each chunk.
"""

import pandas as pd
import numpy as np
import h3
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import os

CHUNK_SIZE = 500_000
SAMPLE_SIZE = 100_000


def compute_split_timestamp(filepath: str, train_ratio: float = 0.8) -> pd.Timestamp:
    """Compute the split timestamp for train/test."""
    print("\nComputing train/test split timestamp...")
    unique_times = set()
    for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE, usecols=['hour'], parse_dates=['hour']):
        unique_times.update(chunk['hour'].unique())
    
    unique_times = sorted(unique_times)
    split_idx = int(len(unique_times) * train_ratio)
    split_time = unique_times[split_idx]
    print(f"  Split timestamp: {split_time}")
    return split_time


def generate_folium_map_chunked(input_filepath: str, output_path: str) -> None:
    """Generate a Folium map from chunked data."""
    print(f"\nGenerating Folium map...")
    
    hex_accidents = {}
    for chunk in pd.read_csv(input_filepath, chunksize=CHUNK_SIZE, usecols=['h3_index', 'accident_count']):
        agg = chunk.groupby('h3_index')['accident_count'].sum()
        for hex_id, count in agg.items():
            hex_accidents[hex_id] = hex_accidents.get(hex_id, 0) + count
    
    print(f"  Total hexagons: {len(hex_accidents)}")
    
    nyc_center = [40.7128, -74.0060]
    m = folium.Map(location=nyc_center, zoom_start=11, tiles='CartoDB positron')
    max_accidents = max(hex_accidents.values())
    
    for hex_id, accidents in hex_accidents.items():
        boundary = h3.cell_to_boundary(hex_id)
        boundary_coords = [[coord[0], coord[1]] for coord in boundary]
        intensity = accidents / max_accidents
        color = f'#{int(255 * intensity):02x}0000'
        folium.Polygon(
            locations=boundary_coords, color=color, fill=True,
            fill_color=color, fill_opacity=0.6, weight=1,
            popup=f"H3: {hex_id}<br>Accidents: {accidents:,}"
        ).add_to(m)
    
    m.save(output_path)
    print(f"  Map saved to: {output_path}")


def generate_correlation_heatmap_sampled(input_filepath: str, output_path: str) -> None:
    """Generate correlation heatmap using fast chunk-based sampling."""
    print(f"\nGenerating correlation heatmap...")
    
    samples = []
    samples_per_chunk = SAMPLE_SIZE // 53
    
    for chunk in pd.read_csv(input_filepath, chunksize=CHUNK_SIZE, parse_dates=['hour']):
        n_sample = min(samples_per_chunk, len(chunk))
        if n_sample > 0:
            samples.append(chunk.sample(n=n_sample, random_state=42))
    
    sample_df = pd.concat(samples, ignore_index=True)
    print(f"  Sampled {len(sample_df):,} rows")
    
    # Add lag features to sample (vectorized)
    sample_df = sample_df.sort_values(['h3_index', 'hour'])
    sample_df['accidents_1h_ago'] = sample_df.groupby('h3_index')['accident_count'].shift(1).fillna(0)
    sample_df['accidents_24h_ago'] = sample_df.groupby('h3_index')['accident_count'].shift(24).fillna(0)
    sample_df['rolling_mean_7d'] = sample_df.groupby('h3_index')['accident_count'].transform(
        lambda x: x.rolling(window=168, min_periods=1).mean()
    )
    
    corr_cols = ['accident_count', 'temperature', 'precipitation', 'wind_speed', 
                 'snow_depth', 'is_holiday', 'is_weekend', 'hour_of_day', 
                 'day_of_week', 'month', 'accidents_1h_ago', 'accidents_24h_ago', 'rolling_mean_7d']
    available_cols = [c for c in corr_cols if c in sample_df.columns]
    
    corr_df = sample_df[available_cols].copy()
    for col in corr_df.columns:
        if corr_df[col].dtype == 'bool':
            corr_df[col] = corr_df[col].astype(int)
    
    corr_matrix = corr_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0, square=True)
    plt.title('Correlation Heatmap: Weather vs Accident Count', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Heatmap saved to: {output_path}")
    print("\n  Top correlations with accident_count:")
    accident_corr = corr_matrix['accident_count'].drop('accident_count').sort_values(key=abs, ascending=False)
    for feature, corr in accident_corr.head(5).items():
        print(f"    {feature}: {corr:.4f}")


def process_and_split_vectorized(
    input_filepath: str, 
    train_filepath: str, 
    test_filepath: str,
    split_time: pd.Timestamp
) -> None:
    """
    Vectorized processing: Add lag features and split into train/test.
    
    Uses pandas vectorized operations within each chunk.
    Lag features are approximate (computed within chunks sorted by hex+time).
    """
    print("\nProcessing data with lag features (vectorized)...")
    
    train_initialized = False
    test_initialized = False
    train_count = 0
    test_count = 0
    chunk_num = 0
    
    for chunk in pd.read_csv(input_filepath, chunksize=CHUNK_SIZE, parse_dates=['hour']):
        chunk_num += 1
        if chunk_num % 5 == 0:
            print(f"  Processing chunk {chunk_num}...")
        
        # Sort by hexagon and time within chunk
        chunk = chunk.sort_values(['h3_index', 'hour'])
        
        # Vectorized lag features (approximate - computed within chunk)
        chunk['accidents_1h_ago'] = chunk.groupby('h3_index')['accident_count'].shift(1).fillna(0)
        chunk['accidents_24h_ago'] = chunk.groupby('h3_index')['accident_count'].shift(24).fillna(0)
        chunk['rolling_mean_7d'] = chunk.groupby('h3_index')['accident_count'].transform(
            lambda x: x.rolling(window=168, min_periods=1).mean()
        )
        
        # Split into train/test
        train_chunk = chunk[chunk['hour'] < split_time]
        test_chunk = chunk[chunk['hour'] >= split_time]
        
        # Write to files
        if len(train_chunk) > 0:
            train_chunk.to_csv(train_filepath, mode='a', header=not train_initialized, index=False)
            train_initialized = True
            train_count += len(train_chunk)
        
        if len(test_chunk) > 0:
            test_chunk.to_csv(test_filepath, mode='a', header=not test_initialized, index=False)
            test_initialized = True
            test_count += len(test_chunk)
    
    print(f"  Processed {chunk_num} chunks")
    print(f"  Train records: {train_count:,}")
    print(f"  Test records: {test_count:,}")


def main():
    input_filepath = 'data/processed/crash_dataset.csv'
    train_filepath = 'data/processed/train.csv'
    test_filepath = 'data/processed/test.csv'
    map_filepath = 'data/processed/nyc_accident_map.html'
    heatmap_filepath = 'data/processed/correlation_heatmap.png'
    
    print("="*60)
    print("NYC Crash - Feature Engineering (Vectorized)")
    print("="*60)
    
    os.makedirs('data/processed', exist_ok=True)
    
    # Remove old output files
    for f in [train_filepath, test_filepath]:
        if os.path.exists(f):
            os.remove(f)
    
    # Step 1: Compute split timestamp
    split_time = compute_split_timestamp(input_filepath)
    
    # Step 2: Generate Folium map
    generate_folium_map_chunked(input_filepath, map_filepath)
    
    # Step 3: Generate correlation heatmap
    generate_correlation_heatmap_sampled(input_filepath, heatmap_filepath)
    
    # Step 4: Vectorized lag features + split
    process_and_split_vectorized(input_filepath, train_filepath, test_filepath, split_time)
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)
    print(f"\nOutput files saved to data/processed/")


if __name__ == '__main__':
    main()
