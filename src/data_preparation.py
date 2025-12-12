"""
Data Preparation Script for NYC Crash Risk Prediction System

This script:
1. Loads crash collision data from CSV
2. Converts Date/Time to a single datetime object
3. Fetches hourly weather data from Meteostat for NYC 
4. Cleans weather data (forward fill imputation)
5. Saves weather data to CSV
"""

import pandas as pd
from datetime import datetime
from meteostat import Hourly


def load_and_process_crashes(filepath: str) -> pd.DataFrame:
    """
    Load collision data and convert Date/Time to single datetime object.
    
    Args:
        filepath: Path to the collisions CSV file
        
    Returns:
        DataFrame with processed crash data including datetime column
    """
    print(f"Loading crash data from {filepath}...")
    
    # Load the CSV
    df = pd.read_csv(filepath)
    
    # Convert CRASH DATE and CRASH TIME to single datetime object
    df['DATETIME'] = pd.to_datetime(
        df['CRASH DATE'] + ' ' + df['CRASH TIME'],
        format='%m/%d/%Y %H:%M'
    )
    
    print(f"Loaded {len(df):,} crash records")
    print(f"Date range: {df['DATETIME'].min()} to {df['DATETIME'].max()}")
    
    return df


def fetch_weather_data(start_date: datetime, end_date: datetime, station_id: str = '72502') -> pd.DataFrame:
    """
    Fetch hourly weather data from Meteostat for NYC.
    
    Args:
        start_date: Start date for weather data
        end_date: End date for weather data
        station_id: Meteostat station ID (default: 72502 for NYC/LaGuardia)
        
    Returns:
        DataFrame with hourly weather data
    """
    print(f"\nFetching weather data for station {station_id}...")
    print(f"Date range: {start_date} to {end_date}")
    
    # Fetch hourly data using the station ID directly
    # Station 72502 is LaGuardia Airport (KLGA) in NYC
    weather = Hourly(station_id, start_date, end_date)
    weather_df = weather.fetch()
    
    if weather_df.empty:
        raise ValueError(f"No weather data found for station {station_id}")
    
    print(f"Fetched {len(weather_df):,} hourly weather records")
    
    return weather_df


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean weather data by selecting relevant columns and imputing missing values.
    
    Args:
        df: Raw weather DataFrame from Meteostat
        
    Returns:
        Cleaned DataFrame with imputed values
    """
    print("\nCleaning weather data...")
    
    # Select relevant columns (temp, precipitation/rain, wind, snow)
    # Meteostat columns: temp, prcp (precipitation), wspd (wind speed), snow
    columns_of_interest = ['temp', 'prcp', 'wspd', 'snow']
    
    # Keep only columns that exist in the data
    available_columns = [col for col in columns_of_interest if col in df.columns]
    weather_clean = df[available_columns].copy()
    
    # Report missing values before imputation
    print("\nMissing values before imputation:")
    for col in available_columns:
        missing = weather_clean[col].isna().sum()
        total = len(weather_clean)
        print(f"  {col}: {missing:,} ({missing/total*100:.1f}%)")
    
    # Forward fill imputation for missing values
    weather_clean = weather_clean.ffill()
    
    # Backward fill any remaining NaN at the start
    weather_clean = weather_clean.bfill()
    
    # Fill any remaining NaN with 0 (for columns that are entirely missing like snow)
    weather_clean = weather_clean.fillna(0)
    
    # Report final status
    remaining_missing = weather_clean.isna().sum().sum()
    print(f"\nRemaining missing values after imputation: {remaining_missing}")
    
    # Rename columns for clarity
    column_mapping = {
        'temp': 'temperature',
        'prcp': 'precipitation',
        'wspd': 'wind_speed',
        'snow': 'snow_depth'
    }
    weather_clean = weather_clean.rename(columns=column_mapping)
    
    return weather_clean


def save_weather_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save weather data to CSV file.
    
    Args:
        df: Weather DataFrame to save
        filepath: Output file path
    """
    print(f"\nSaving weather data to {filepath}...")
    df.to_csv(filepath, index=True)  # index=True to preserve datetime index
    print(f"Saved {len(df):,} records")


def main():
    """Main execution function."""
    # File paths
    input_filepath = 'data/raw/collisions.csv'
    output_filepath = 'data/raw/weather_history.csv'
    
    # Station ID for NYC weather (LaGuardia Airport)
    station_id = '72502'
    
    # Step 1: Load and process crash data
    crashes_df = load_and_process_crashes(input_filepath)
    
    # Step 2: Get date range from crash data
    start_date = crashes_df['DATETIME'].min().to_pydatetime()
    end_date = crashes_df['DATETIME'].max().to_pydatetime()
    
    # Step 3: Fetch weather data for the exact date range
    weather_df = fetch_weather_data(start_date, end_date, station_id)
    
    # Step 4: Clean weather data (impute missing values)
    weather_clean = clean_weather_data(weather_df)
    
    # Step 5: Save weather data
    save_weather_data(weather_clean, output_filepath)
    
    print("\n" + "="*50)
    print("Data preparation complete!")
    print("="*50)
    
    # Display summary
    print("\nWeather Data Summary:")
    print(weather_clean.describe())


if __name__ == '__main__':
    main()
