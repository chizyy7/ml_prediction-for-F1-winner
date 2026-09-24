"""
Data cleaning module for F1 Championship Predictor.
Handles validation and cleaning of raw F1 data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1DataCleaner:
    """Cleans and validates F1 data."""

    def __init__(self):
        self.valid_statuses = [
            'Finished', '+1 Lap', '+2 Laps', '+3 Laps', '+4 Laps', '+5 Laps',
            '+6 Laps', '+7 Laps', '+8 Laps', '+9 Laps', '+10 Laps',
            'Accident', 'Collision', 'Engine', 'Gearbox', 'Hydraulics',
            'Electrical', 'Brakes', 'Spyker', 'Transmission', 'Clutch',
            'Suspension', 'Wheel', 'Puncture', 'Fire', 'Withdrew'
        ]

    def clean_race_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean race results DataFrame."""
        if df.empty:
            return df

        logger.info("Cleaning race results...")
        df_clean = df.copy()

        # Ensure numeric columns are proper types
        numeric_cols = ['year', 'round', 'grid_position', 'finish_position',
                       'points', 'laps', 'milliseconds']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Validate finish positions (should be 1-20 for typical F1)
        if 'finish_position' in df_clean.columns:
            # Mark invalid positions (non-numeric values like 'R', 'W', etc. are already NaN)
            invalid_pos = (~df_clean['finish_position'].between(1, 20, inclusive='both')) & df_clean['finish_position'].notna()
            if invalid_pos.any():
                logger.warning(f"Found {invalid_pos.sum()} invalid finish positions, setting to NaN")
                df_clean.loc[invalid_pos, 'finish_position'] = np.nan

        # Validate grid positions
        if 'grid_position' in df_clean.columns:
            invalid_grid = (~df_clean['grid_position'].between(1, 20, inclusive='both')) & df_clean['grid_position'].notna()
            if invalid_grid.any():
                logger.warning(f"Found {invalid_grid.sum()} invalid grid positions, setting to NaN")
                df_clean.loc[invalid_grid, 'grid_position'] = np.nan

        # Validate points (should be non-negative)
        if 'points' in df_clean.columns:
            invalid_points = df_clean['points'] < 0
            if invalid_points.any():
                logger.warning(f"Found {invalid_points.sum()} negative points, setting to 0")
                df_clean.loc[invalid_points, 'points'] = 0

        # Validate laps (should be positive if finished)
        if 'laps' in df_clean.columns:
            invalid_laps = (df_clean['laps'] < 0) & df_clean['laps'].notna()
            if invalid_labs.any():
                logger.warning(f"Found {invalid_labs.sum()} negative laps, setting to NaN")
                df_clean.loc[invalid_labs, 'laps'] = np.nan

        # Clean status field
        if 'status' in df_clean.columns:
            # Standardize common status values
            status_mapping = {
                'Accident': 'Accident',
                'Collison damage': 'Collision',
                'Collision': 'Collision',
                'Collision damage': 'Collision',
                'Engine': 'Engine',
                'Gearbox': 'Gearbox',
                'Hydraulic': 'Hydraulics',
                'Hydraulics': 'Hydraulics',
                'Electrical': 'Electrical',
                'Energy store': 'Electrical',
                'Brakes': 'Brakes',
                'Clutch': 'Clutch',
                'Suspension': 'Suspension',
                'Wheel': 'Wheel',
                'Puncture': 'Puncture',
                'Fire': 'Fire',
                'Withdrawn': 'Withdrew',
                'Withdrew': 'Withdrew'
            }

            # Apply mapping where possible
            def map_status(status):
                if pd.isna(status):
                    return status
                status_str = str(status).strip()
                # Check for exact matches first
                if status_str in self.valid_statuses:
                    return status_str
                # Check for partial matches
                for key, value in status_mapping.items():
                    if key.lower() in status_str.lower():
                        return value
                # Default to original if no match
                return status_str

            df_clean['status'] = df_clean['status'].apply(map_status)

        # Calculate derived fields
        if 'grid_position' in df_clean.columns and 'finish_position' in df_clean.columns:
            # Positions gained (negative means lost positions)
            df_clean['positions_gained'] = df_clean['grid_position'] - df_clean['finish_position']
            # Only calculate for finished races where both values exist
            mask = df_clean['grid_position'].notna() & df_clean['finish_position'].notna()
            df_clean.loc[~mask, 'positions_gained'] = np.nan

        # Did Not Finish flag
        if 'status' in df_clean.columns:
            df_clean['dnf'] = ~df_clean['status'].isin(['Finished'] + [f'+{i} Laps' for i in range(1, 11)])
            # Treat lapped finishes as finished (they completed the race)
            df_clean.loc[df_clean['status'].str.match(r'^\+\d+ Laps$', na=False), 'dnf'] = False

        logger.info(f"Cleaned {len(df_clean)} race results")
        return df_clean

    def clean_qualifying_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean qualifying results DataFrame."""
        if df.empty:
            return df

        logger.info("Cleaning qualifying results...")
        df_clean = df.copy()

        # Ensure numeric columns
        numeric_cols = ['year', 'round', 'position']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Validate qualifying positions (1-20 typically)
        if 'position' in df_clean.columns:
            invalid_pos = (~df_clean['position'].between(1, 20, inclusive='both')) & df_clean['position'].notna()
            if invalid_pos.any():
                logger.warning(f"Found {invalid_pos.sum()} invalid qualifying positions, setting to NaN")
                df_clean.loc[invalid_pos, 'position'] = np.nan

        # Clean time fields (Q1, Q2, Q3) - convert to seconds for easier processing
        for q_col in ['q1', 'q2', 'q3']:
            if q_col in df_clean.columns:
                df_clean[f'{q_col}_seconds'] = df_clean[q_col].apply(self._time_to_seconds)

        logger.info(f"Cleaned {len(df_clean)} qualifying results")
        return df_clean

    def clean_driver_standings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean driver standings DataFrame."""
        if df.empty:
            return df

        logger.info("Cleaning driver standings...")
        df_clean = df.copy()

        # Ensure numeric columns
        numeric_cols = ['year', 'points', 'position', 'wins']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Validate points (non-negative)
        if 'points' in df_clean.columns:
            invalid_points = df_clean['points'] < 0
            if invalid_points.any():
                logger.warning(f"Found {invalid_points.sum()} negative points, setting to 0")
                df_clean.loc[invalid_points, 'points'] = 0

        # Validate position (1-20 typically)
        if 'position' in df_clean.columns:
            invalid_pos = (~df_clean['position'].between(1, 20, inclusive='both')) & df_clean['position'].notna()
            if invalid_pos.any():
                logger.warning(f"Found {invalid_pos.sum()} invalid standings positions, setting to NaN")
                df_clean.loc[invalid_pos, 'position'] = np.nan

        logger.info(f"Cleaned {len(df_clean)} driver standings")
        return df_clean

    def clean_constructor_standings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean constructor standings DataFrame."""
        if df.empty:
            return df

        logger.info("Cleaning constructor standings...")
        df_clean = df.copy()

        # Ensure numeric columns
        numeric_cols = ['year', 'points', 'position', 'wins']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Validate points (non-negative)
        if 'points' in df_clean.columns:
            invalid_points = df_clean['points'] < 0
            if invalid_points.any():
                logger.warning(f"Found {invalid_points.sum()} negative points, setting to 0")
                df_clean.loc[invalid_points, 'points'] = 0

        # Validate position (typically 1-10 for constructors)
        if 'position' in df_clean.columns:
            invalid_pos = (~df_clean['position'].between(1, 10, inclusive='both')) & df_clean['position'].notna()
            if invalid_pos.any():
                logger.warning(f"Found {invalid_pos.sum()} invalid constructor positions, setting to NaN")
                df_clean.loc[invalid_pos, 'position'] = np.nan

        logger.info(f"Cleaned {len(df_clean)} constructor standings")
        return df_clean

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert time string (mm:ss.sss or ss.sss) to seconds."""
        if pd.isna(time_str) or time_str == '' or time_str is None:
            return np.nan

        try:
            # Handle format like "1:23.456" or "89.012"
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                if len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                else:
                    logger.warning(f"Unexpected time format: {time_str}")
                    return np.nan
            else:
                # Just seconds
                return float(time_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse time: {time_str}")
            return np.nan

    def validate_data_integrity(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Validate data integrity and return issues found."""
        issues = {}

        for name, df in data_dict.items():
            if df.empty:
                issues[name] = ["DataFrame is empty"]
                continue

            df_issues = []

            # Check for completely null columns
            null_cols = df.columns[df.isnull().all()].tolist()
            if null_cols:
                df_issues.append(f"Columns with all null values: {null_cols}")

            # Check for duplicate races (year, round combination)
            if 'year' in df.columns and 'round' in df.columns:
                dup_races = df.duplicated(subset=['year', 'round'], keep=False)
                if dup_races.any():
                    dup_count = dup_races.sum()
                    df_issues.append(f"Found {dup_count} duplicate race entries")

            # Check for impossible values
            if 'finish_position' in df.columns:
                impossible_finish = df['finish_position'].between(21, 999, inclusive='both')
                if impossible_finish.any():
                    df_issues.append(f"Found {impossible_finish.sum()} impossible finish positions (>20)")

            if 'grid_position' in df.columns:
                impossible_grid = df['grid_position'].between(21, 999, inclusive='both')
                if impossible_grid.any():
                    df_issues.append(f"Found {impossible_grid.sum()} impossible grid positions (>20)")

            issues[name] = df_issues if df_issues else ["No issues found"]

        return issues

def clean_all_data(raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed") -> None:
    """Clean all raw data and save to processed directory."""
    import os
    import glob

    # Create processed directory
    os.makedirs(processed_data_dir, exist_ok=True)

    cleaner = F1DataCleaner()

    # Process race results
    race_files = glob.glob(f"{raw_data_dir}/results_*.parquet")
    for file_path in race_files:
        try:
            df = pd.read_parquet(file_path)
            df_clean = cleaner.clean_race_results(df)

            # Save cleaned data
            filename = os.path.basename(file_path)
            clean_path = f"{processed_data_dir}/clean_{filename}"
            df_clean.to_parquet(clean_path, index=False)

            logger.info(f"Cleaned and saved {filename}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Process qualifying results
    quali_files = glob.glob(f"{raw_data_dir}/qualifying_*.parquet")
    for file_path in quali_files:
        try:
            df = pd.read_parquet(file_path)
            df_clean = cleaner.clean_qualifying_results(df)

            # Save cleaned data
            filename = os.path.basename(file_path)
            clean_path = f"{processed_data_dir}/clean_{filename}"
            df_clean.to_parquet(clean_path, index=False)

            logger.info(f"Cleaned and saved {filename}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Process driver standings
    driver_standings_files = glob.glob(f"{raw_data_dir}/driver_standings_*.parquet")
    for file_path in driver_standings_files:
        try:
            df = pd.read_parquet(file_path)
            df_clean = cleaner.clean_driver_standings(df)

            # Save cleaned data
            filename = os.path.basename(file_path)
            clean_path = f"{processed_data_dir}/clean_{filename}"
            df_clean.to_parquet(clean_path, index=False)

            logger.info(f"Cleaned and saved {filename}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Process constructor standings
    constructor_standings_files = glob.glob(f"{raw_data_dir}/constructor_standings_*.parquet")
    for file_path in constructor_standings_files:
        try:
            df = pd.read_parquet(file_path)
            df_clean = cleaner.clean_constructor_standings(df)

            # Save cleaned data
            filename = os.path.basename(file_path)
            clean_path = f"{processed_data_dir}/clean_{filename}"
            df_clean.to_parquet(clean_path, index=False)

            logger.info(f"Cleaned and saved {filename}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Process races and circuits (less cleaning needed)
    for entity in ['races', 'circuits']:
        files = glob.glob(f"{raw_data_dir}/{entity}*.parquet")
        for file_path in files:
            try:
                df = pd.read_parquet(file_path)
                # Basic cleaning - ensure proper types
                df_clean = df.copy()

                # Save cleaned data
                filename = os.path.basename(file_path)
                clean_path = f"{processed_data_dir}/clean_{filename}"
                df_clean.to_parquet(clean_path, index=False)

                logger.info(f"Cleaned and saved {filename}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

def main():
    """Main function for running data cleaning."""
    clean_all_data()
    print("Data cleaning complete!")

if __name__ == "__main__":
    main()