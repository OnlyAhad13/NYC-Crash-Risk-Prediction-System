"""
Spatial Binning Script for NYC Crash Risk Prediction System

This script:
1. Converts crash coordinates to H3 hexagon indices
2. Aggregates crashes by hexagon and hour
3. Creates a master frame with all hexagon-hour combinations
4. Merges with weather data
5. Adds holiday information
"""

import pandas as pd
import numpy as np
import h3
import holidays
from datetime import datetime
from typing import Tuple


def convert_to_h3_index(df: pd.DataFrame, resolution: int = 7) -> pd.DataFrame:
    """
    Convert LATITUDE and LONGITUDE to H3 hexagon index.
    
    Args:
        df: DataFrame with LATITUDE and LONGITUDE columns
        resolution: H3 resolution level (7 or 8 recommended)
        
    Returns:
        DataFrame with added h3_index column
    """
    print(f"\nConverting coordinates to H3 indices (resolution {resolution})...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Filter out rows with missing or invalid coordinates
    valid_coords_mask = (
        df['LATITUDE'].notna() & 
        df['LONGITUDE'].notna() &
        (df['LATITUDE'] != 0) & 
        (df['LONGITUDE'] != 0) &
        (df['LATITUDE'].between(-90, 90)) &
        (df['LONGITUDE'].between(-180, 180))
    )
    
    invalid_count = (~valid_coords_mask).sum()
    print(f"  Rows with invalid/missing coordinates: {invalid_count:,} ({invalid_count/len(df)*100:.1f}%)")
    
    # Initialize h3_index column with None
    df['h3_index'] = None
    
    # Convert valid coordinates to H3 indices
    valid_df = df.loc[valid_coords_mask]
    
    # Apply H3 conversion using latlng_to_cell (h3 v4.x API)
    df.loc[valid_coords_mask, 'h3_index'] = valid_df.apply(
        lambda row: h3.latlng_to_cell(row['LATITUDE'], row['LONGITUDE'], resolution),
        axis=1
    )
    
    # Filter to only valid H3 indices
    df = df[df['h3_index'].notna()].copy()
    
    unique_hexagons = df['h3_index'].nunique()
    print(f"  Unique hexagons created: {unique_hexagons:,}")
    print(f"  Valid crash records with H3 index: {len(df):,}")
    
    return df


def aggregate_crashes_by_hexagon_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate crash data by H3 hexagon and hour.
    
    Args:
        df: DataFrame with h3_index and DATETIME columns
        
    Returns:
        Aggregated DataFrame with accident counts per hexagon per hour
    """
    print("\nAggregating crashes by hexagon and hour...")
    
    # Create hour column (floor to nearest hour)
    df = df.copy()
    df['hour'] = df['DATETIME'].dt.floor('h')
    
    # Aggregate by h3_index and hour
    aggregated = df.groupby(['h3_index', 'hour']).agg(
        accident_count=('DATETIME', 'count'),
        total_injured=('NUMBER OF PERSONS INJURED', 'sum')
    ).reset_index()
    
    print(f"  Aggregated records (hexagon-hour pairs with crashes): {len(aggregated):,}")
    print(f"  Date range: {aggregated['hour'].min()} to {aggregated['hour'].max()}")
    
    return aggregated


def create_master_frame(
    unique_hexagons: pd.Series, 
    start_time: datetime, 
    end_time: datetime
) -> pd.DataFrame:
    """
    Create a master frame with all hexagon-hour combinations.
    
    This creates a complete time series for every unique hexagon,
    for every hour in the dataset range.
    
    Args:
        unique_hexagons: Series of unique H3 indices
        start_time: Start datetime (will be floored to hour)
        end_time: End datetime (will be ceiled to hour)
        
    Returns:
        Master DataFrame with all hexagon-hour combinations
    """
    print("\nCreating master frame with all hexagon-hour combinations...")
    
    # Floor start and ceil end to nearest hour
    start_hour = pd.Timestamp(start_time).floor('h')
    end_hour = pd.Timestamp(end_time).ceil('h')
    
    # Create hourly date range
    hourly_range = pd.date_range(start=start_hour, end=end_hour, freq='h')
    print(f"  Hours in range: {len(hourly_range):,}")
    print(f"  Unique hexagons: {len(unique_hexagons):,}")
    
    # Calculate expected size
    expected_rows = len(hourly_range) * len(unique_hexagons)
    print(f"  Expected master frame size: {expected_rows:,} rows")
    
    # Create the cartesian product of hexagons and hours
    master_frame = pd.DataFrame({
        'h3_index': np.repeat(unique_hexagons.values, len(hourly_range)),
        'hour': np.tile(hourly_range, len(unique_hexagons))
    })
    
    print(f"  Master frame created: {len(master_frame):,} rows")
    
    return master_frame


def merge_with_crash_counts(
    master_frame: pd.DataFrame, 
    crash_counts: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge master frame with crash counts, filling missing with 0.
    
    Args:
        master_frame: Complete hexagon-hour combinations
        crash_counts: Aggregated crash counts
        
    Returns:
        Merged DataFrame with accident_count filled with 0 for no crashes
    """
    print("\nMerging master frame with crash counts...")
    
    # Merge on h3_index and hour
    merged = master_frame.merge(
        crash_counts,
        on=['h3_index', 'hour'],
        how='left'
    )
    
    # Fill NaN accident counts with 0
    merged['accident_count'] = merged['accident_count'].fillna(0).astype(int)
    merged['total_injured'] = merged['total_injured'].fillna(0).astype(int)
    
    # Stats
    non_zero = (merged['accident_count'] > 0).sum()
    print(f"  Total records: {len(merged):,}")
    print(f"  Records with accidents: {non_zero:,} ({non_zero/len(merged)*100:.2f}%)")
    print(f"  Records without accidents: {len(merged) - non_zero:,}")
    
    return merged


def merge_with_weather(df: pd.DataFrame, weather_filepath: str) -> pd.DataFrame:
    """
    Merge dataset with weather data based on timestamp.
    
    Args:
        df: DataFrame with 'hour' column
        weather_filepath: Path to weather CSV file
        
    Returns:
        DataFrame merged with weather data
    """
    print(f"\nMerging with weather data from {weather_filepath}...")
    
    # Load weather data
    weather = pd.read_csv(weather_filepath, parse_dates=['time'], index_col='time')
    weather = weather.reset_index().rename(columns={'time': 'hour'})
    
    # Ensure hour columns are same type
    df['hour'] = pd.to_datetime(df['hour'])
    weather['hour'] = pd.to_datetime(weather['hour'])
    
    # Merge
    merged = df.merge(weather, on='hour', how='left')
    
    # Check for missing weather data
    missing_weather = merged['temperature'].isna().sum()
    if missing_weather > 0:
        print(f"  Warning: {missing_weather:,} records missing weather data")
        # Forward fill then backward fill missing weather data
        weather_cols = ['temperature', 'precipitation', 'wind_speed', 'snow_depth']
        for col in weather_cols:
            merged[col] = merged[col].ffill().bfill().fillna(0)
    
    print(f"  Successfully merged weather data")
    
    return merged


def add_holiday_column(df: pd.DataFrame, country: str = 'US', state: str = 'NY') -> pd.DataFrame:
    """
    Add boolean is_holiday column based on US/NY holidays.
    
    Args:
        df: DataFrame with 'hour' column
        country: Country code for holidays
        state: State code for holidays
        
    Returns:
        DataFrame with is_holiday column
    """
    print("\nAdding holiday information...")
    
    # Get unique years in the data
    years = df['hour'].dt.year.unique()
    print(f"  Years covered: {years.min()} to {years.max()}")
    
    # Create holiday calendar for all years
    us_holidays = holidays.country_holidays(country, subdiv=state, years=list(years))
    
    # Add is_holiday column
    df = df.copy()
    df['date'] = df['hour'].dt.date
    df['is_holiday'] = df['date'].apply(lambda x: x in us_holidays)
    df = df.drop(columns=['date'])
    
    # Stats
    holiday_hours = df['is_holiday'].sum()
    print(f"  Holiday hours: {holiday_hours:,} ({holiday_hours/len(df)*100:.2f}%)")
    
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional temporal features useful for prediction.
    
    Args:
        df: DataFrame with 'hour' column
        
    Returns:
        DataFrame with additional temporal features
    """
    print("\nAdding temporal features...")
    
    df = df.copy()
    df['hour_of_day'] = df['hour'].dt.hour
    df['day_of_week'] = df['hour'].dt.dayofweek
    df['month'] = df['hour'].dt.month
    df['year'] = df['hour'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    print("  Added: hour_of_day, day_of_week, month, year, is_weekend")
    
    return df


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save processed dataset to CSV.
    
    Args:
        df: Processed DataFrame
        filepath: Output file path
    """
    print(f"\nSaving processed data to {filepath}...")
    df.to_csv(filepath, index=False)
    print(f"  Saved {len(df):,} records")


def main():
    """Main execution function for spatial binning pipeline."""
    from data_preparation import load_and_process_crashes
    
    # Configuration
    H3_RESOLUTION = 7  # Resolution 7 gives ~5km² hexagons
    
    # File paths
    crashes_filepath = 'data/raw/collisions.csv'
    weather_filepath = 'data/raw/weather_history.csv'
    output_filepath = 'data/processed/crash_dataset.csv'
    
    print("="*60)
    print("NYC Crash Risk Prediction - Spatial Binning Pipeline")
    print("="*60)
    
    # Step 1: Load crash data
    crashes_df = load_and_process_crashes(crashes_filepath)
    
    # Step 2: Convert to H3 indices
    crashes_h3 = convert_to_h3_index(crashes_df, resolution=H3_RESOLUTION)
    
    # Step 3: Aggregate by hexagon and hour
    crash_counts = aggregate_crashes_by_hexagon_hour(crashes_h3)
    
    # Step 4: Get unique hexagons and time range
    unique_hexagons = crashes_h3['h3_index'].unique()
    unique_hexagons = pd.Series(unique_hexagons)
    
    start_time = crashes_h3['DATETIME'].min()
    end_time = crashes_h3['DATETIME'].max()
    
    # Step 5: Create master frame
    master_frame = create_master_frame(unique_hexagons, start_time, end_time)
    
    # Step 6: Merge with crash counts
    dataset = merge_with_crash_counts(master_frame, crash_counts)
    
    # Step 7: Merge with weather data
    dataset = merge_with_weather(dataset, weather_filepath)
    
    # Step 8: Add holiday information
    dataset = add_holiday_column(dataset)
    
    # Step 9: Add temporal features
    dataset = add_temporal_features(dataset)
    
    # Step 10: Save processed data
    import os
    os.makedirs('data/processed', exist_ok=True)
    save_processed_data(dataset, output_filepath)
    
    # Final summary
    print("\n" + "="*60)
    print("Spatial Binning Pipeline Complete!")
    print("="*60)
    
    print("\nDataset Summary:")
    print(f"  Total records: {len(dataset):,}")
    print(f"  Unique hexagons: {dataset['h3_index'].nunique():,}")
    print(f"  Date range: {dataset['hour'].min()} to {dataset['hour'].max()}")
    print(f"  Total accidents: {dataset['accident_count'].sum():,}")
    
    print("\nColumn list:")
    print(f"  {list(dataset.columns)}")
    
    print("\nDataset preview:")
    print(dataset.head(10))
    
    print("\nAccident count distribution:")
    print(dataset['accident_count'].describe())


if __name__ == '__main__':
    main()
