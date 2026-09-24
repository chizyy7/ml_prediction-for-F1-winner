"""
Feature engineering module for F1 Championship Predictor.
Creates meaningful features for ML models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1FeatureEngineer:
    """Engineers features for F1 championship prediction."""

    def __init__(self):
        self.lewis_adjustment_factor = 0.95  # Placeholder for driver-specific adjustments

    def create_driver_features(self,
                             races_df: pd.DataFrame,
                             results_df: pd.DataFrame,
                             qualifying_df: pd.DataFrame,
                             driver_standings_df: pd.DataFrame) -> pd.DataFrame:
        """Create driver-level features for each race."""
        logger.info("Creating driver features...")

        if results_df.empty:
            return pd.DataFrame()

        # Start with basic race results
        features_df = results_df[['year', 'round', 'driver_id', 'constructor_id']].copy()
        features_df = features_df.sort_values(['driver_id', 'year', 'round']).reset_index(drop=True)

        # Merge with qualifying data
        if not qualifying_df.empty:
            quali_features = qualifying_df[['year', 'round', 'driver_id', 'position', 'q1_seconds', 'q2_seconds', 'q3_seconds']].copy()
            quali_features = quali_features.rename(columns={
                'position': 'quali_position',
                'q1_seconds': 'q1_time',
                'q2_seconds': 'q2_time',
                'q3_seconds': 'q3_time'
            })
            features_df = features_df.merge(quali_features, on=['year', 'round', 'driver_id'], how='left')

        # Calculate driver statistics up to each race (excluding current race)
        driver_features_list = []

        for driver_id in features_df['driver_id'].unique():
            driver_races = features_df[features_df['driver_id'] == driver_id].copy()
            driver_results = results_df[results_df['driver_id'] == driver_id].copy()
            driver_quali = qualifying_df[qualifying_df['driver_id'] == driver_id].copy() if not qualifying_df.empty else pd.DataFrame()

            # Sort by year and round to ensure chronological order
            driver_races = driver_races.sort_values(['year', 'round']).reset_index(drop=True)
            driver_results = driver_results.sort_values(['year', 'round']).reset_index(drop=True)
            if not driver_quali.empty:
                driver_quali = driver_quali.sort_values(['year', 'round']).reset_index(drop=True)

            for idx, race in driver_races.iterrows():
                year, round_num = race['year'], race['round']

                # Get historical data (races before current race)
                hist_mask = ((driver_results['year'] < year) |
                           ((driver_results['year'] == year) & (driver_results['round'] < round_num)))
                hist_results = driver_results[hist_mask].copy()

                hist_quali_mask = ((driver_quali['year'] < year) |
                                 ((driver_quali['year'] == year) & (driver_quali['round'] < round_num))) if not driver_quali.empty else []
                hist_quali = driver_quali[hist_quali_mask].copy() if not driver_quali.empty else pd.DataFrame()

                # Calculate features
                features = {
                    'year': year,
                    'round': round_num,
                    'driver_id': driver_id,
                    'constructor_id': race['constructor_id']
                }

                # Basic career stats
                features['races_entered'] = len(hist_results)
                features['career_points'] = hist_results['points'].sum() if len(hist_results) > 0 else 0
                features['career_avg_finish'] = hist_results['finish_position'].mean() if len(hist_results) > 0 else np.nan
                features['career_median_finish'] = hist_results['finish_position'].median() if len(hist_results) > 0 else np.nan
                features['career_win_rate'] = (hist_results['position'] == 1).sum() / len(hist_results) if len(hist_results) > 0 else 0
                features['career_podium_rate'] = (hist_results['position'] <= 3).sum() / len(hist_results) if len(hist_results) > 0 else 0
                features['career_top5_rate'] = (hist_results['position'] <= 5).sum() / len(hist_results) if len(hist_results) > 0 else 0
                features['career_top10_rate'] = (hist_results['position'] <= 10).sum() / len(hist_results) if len(hist_results) > 0 else 0
                features['career_dnf_rate'] = hist_results['dnf'].mean() if len(hist_results) > 0 and 'dnf' in hist_results.columns else 0

                # Qualifying stats
                if not hist_quali.empty:
                    features['career_quali_avg'] = hist_quali['quali_position'].mean() if len(hist_quali) > 0 else np.nan
                    features['career_quali_best'] = hist_quali['quali_position'].min() if len(hist_quali) > 0 else np.nan
                    features['career_pole_rate'] = (hist_quali['quali_position'] == 1).sum() / len(hist_quali) if len(hist_quali) > 0 else 0
                    features['career_q3_rate'] = hist_quali['q3_time'].notna().sum() / len(hist_quali) if len(hist_quali) > 0 else 0
                else:
                    features['career_quali_avg'] = np.nan
                    features['career_quali_best'] = np.nan
                    features['career_pole_rate'] = 0
                    features['career_q3_rate'] = 0

                # Recent form (last 3, 5, 10 races)
                for window in [3, 5, 10]:
                    recent_results = hist_results.tail(window) if len(hist_results) >= window else hist_results
                    recent_quali = hist_quali.tail(window) if len(hist_quali) >= window else hist_quali if not hist_quali.empty else pd.DataFrame()

                    if len(recent_results) > 0:
                        features[f'recent_{window}_avg_finish'] = recent_results['finish_position'].mean()
                        features[f'recent_{window}_points_per_race'] = recent_results['points'].mean()
                        features[f'recent_{window}_win_rate'] = (recent_results['position'] == 1).sum() / len(recent_results)
                        features[f'recent_{window}_podium_rate'] = (recent_results['position'] <= 3).sum() / len(recent_results)
                        features[f'recent_{window}_dnf_rate'] = recent_results['dnf'].mean() if 'dnf' in recent_results.columns else 0

                        # Consistency (std dev of finish positions)
                        features[f'recent_{window}_finish_consistency'] = recent_results['finish_position'].std() if len(recent_results) > 1 else 0
                    else:
                        features[f'recent_{window}_avg_finish'] = np.nan
                        features[f'recent_{window}_points_per_race'] = 0
                        features[f'recent_{window}_win_rate'] = 0
                        features[f'recent_{window}_podium_rate'] = 0
                        features[f'recent_{window}_dnf_rate'] = 0
                        features[f'recent_{window}_finish_consistency'] = 0

                    if not recent_quali.empty and len(recent_quali) > 0:
                        features[f'recent_{window}_avg_quali'] = recent_quali['quali_position'].mean()
                        features[f'recent_{window}_quali_consistency'] = recent_quali['quali_position'].std() if len(recent_quali) > 1 else 0
                    else:
                        features[f'recent_{window}_avg_quali'] = np.nan
                        features[f'recent_{window}_quali_consistency'] = 0

                # Exponentially weighted recent performance (more weight to recent races)
                if len(hist_results) > 0:
                    # Create weights: more recent races get higher weights
                    weights = np.exp(np.arange(len(hist_results)) / len(hist_results))
                    weights = weights / weights.sum()  # Normalize

                    # Weighted average finish position
                    if 'finish_position' in hist_results.columns and not hist_results['finish_position'].isna().all():
                        weighted_finish = np.average(hist_results['finish_position'], weights=weights)
                        features['ewm_avg_finish'] = weighted_finish
                    else:
                        features['ewm_avg_finish'] = np.nan

                    # Weighted points
                    if 'points' in hist_results.columns:
                        weighted_points = np.average(hist_results['points'], weights=weights)
                        features['ewm_avg_points'] = weighted_points
                    else:
                        features['ewm_avg_points'] = 0
                else:
                    features['ewm_avg_finish'] = np.nan
                    features['ewm_avg_points'] = 0

                # Season-to-date stats (current season only)
                std_mask = ((driver_results['year'] == year) & (driver_results['round'] < round_num))
                std_results = driver_results[std_mask].copy()
                std_quali_mask = ((driver_quali['year'] == year) & (driver_quali['round'] < round_num)) if not driver_quali.empty else []
                std_quali = driver_quali[std_quali_mask].copy() if not driver_quali.empty else pd.DataFrame()

                if len(std_results) > 0:
                    features['season_points'] = std_results['points'].sum()
                    features['season_avg_finish'] = std_results['finish_position'].mean()
                    features['season_avg_quali'] = std_quali['quali_position'].mean() if len(str_quali) > 0 else np.nan
                    features['season_wins'] = (std_results['position'] == 1).sum()
                    features['season_podiums'] = (std_results['position'] <= 3).sum()
                    features['season_dnf_rate'] = std_results['dnf'].mean() if 'dnf' in str_results.columns else 0
                else:
                    features['season_points'] = 0
                    features['season_avg_finish'] = np.nan
                    features['season_avg_quali'] = np.nan
                    features['season_wins'] = 0
                    features['season_podiums'] = 0
                    features['season_dnf_rate'] = 0

                # Current qualifying (if available)
                if not qualifying_df.empty:
                    current_quali = qualifying_df[
                        (qualifying_df['year'] == year) &
                        (qualifying_df['round'] == round_num) &
                        (qualifying_df['driver_id'] == driver_id)
                    ]
                    if not current_quali.empty:
                        features['quali_position'] = current_quali.iloc[0]['quali_position']
                        features['q1_time'] = current_quali.iloc[0]['q1_time']
                        features['q2_time'] = current_quali.iloc[0]['q2_time']
                        features['q3_time'] = current_quali.iloc[0]['q3_time']
                    else:
                        features['quali_position'] = np.nan
                        features['q1_time'] = np.nan
                        features['q2_time'] = np.nan
                        features['q3_time'] = np.nan
                else:
                    features['quali_position'] = np.nan
                    features['q1_time'] = np.nan
                    features['q2_time'] = np.nan
                    features['q3_time'] = np.nan

                # Current race results (target variables - for training only)
                current_result = driver_results[
                    (driver_results['year'] == year) &
                    (driver_results['round'] == round_num)
                ]
                if not current_result.empty:
                    features['actual_finish_position'] = current_result.iloc[0]['finish_position']
                    features['actual_points'] = current_result.iloc[0]['points']
                    features['actual_dnf'] = current_result.iloc[0]['dnf'] if 'dnf' in current_result.columns else False
                    features['actual_positions_gained'] = current_result.iloc[0]['positions_gained'] if 'positions_gained' in current_result.columns else np.nan
                else:
                    features['actual_finish_position'] = np.nan
                    features['actual_points'] = np.nan
                    features['actual_dnf'] = np.nan
                    features['actual_positions_gained'] = np.nan

                driver_features_list.append(features)

        driver_features_df = pd.DataFrame(driver_features_list)
        logger.info(f"Created driver features for {len(driver_features_df)} driver-race combinations")
        return driver_features_df

    def create_constructor_features(self,
                                  races_df: pd.DataFrame,
                                  results_df: pd.DataFrame,
                                  constructor_standings_df: pd.DataFrame) -> pd.DataFrame:
        """Create constructor/team-level features."""
        logger.info("Creating constructor features...")

        if results_df.empty:
            return pd.DataFrame()

        features_df = results_df[['year', 'round', 'constructor_id']].copy()
        features_df = features_df.drop_duplicates().sort_values(['constructor_id', 'year', 'round']).reset_index(drop=True)

        constructor_features_list = []

        for constructor_id in features_df['constructor_id'].unique():
            constructor_races = features_df[features_df['constructor_id'] == constructor_id].copy()
            constructor_results = results_df[results_df['constructor_id'] == constructor_id].copy()

            constructor_races = constructor_races.sort_values(['year', 'round']).reset_index(drop=True)
            constructor_results = constructor_results.sort_values(['year', 'round']).reset_index(drop=True)

            for idx, race in constructor_races.iterrows():
                year, round_num = race['year'], race['round']

                # Historical data (before current race)
                hist_mask = ((constructor_results['year'] < year) |
                           ((constructor_results['year'] == year) & (constructor_results['round'] < round_num)))
                hist_results = constructor_results[hist_mask].copy()

                features = {
                    'year': year,
                    'round': round_num,
                    'constructor_id': constructor_id
                }

                # Constructor career stats
                features['const_races_entered'] = len(hist_results)
                features['const_career_points'] = hist_results['points'].sum() if len(hist_results) > 0 else 0
                features['const_career_avg_finish'] = hist_results['finish_position'].mean() if len(hist_results) > 0 else np.nan
                features['const_win_rate'] = (hist_results['position'] == 1).sum() / len(hist_results) if len(hist_results) > 0 else 0
                features['const_podium_rate'] = (hist_results['position'] <= 3).sum() / len(hist_results) if len(hist_results) > 0 else 0
                features['const_dnf_rate'] = hist_results['dnf'].mean() if len(hist_results) > 0 and 'dnf' in hist_results.columns else 0

                # Recent form (last 5 races)
                recent_results = hist_results.tail(5) if len(hist_results) >= 5 else hist_results
                if len(recent_results) > 0:
                    features['const_recent_5_avg_finish'] = recent_results['finish_position'].mean()
                    features['const_recent_5_points_per_race'] = recent_results['points'].mean()
                    features['const_recent_5_dnf_rate'] = recent_results['dnf'].mean() if 'dnf' in recent_results.columns else 0
                else:
                    features['const_recent_5_avg_finish'] = np.nan
                    features['const_recent_5_points_per_race'] = 0
                    features['const_recent_5_dnf_rate'] = 0

                # Season-to-date constructor performance
                std_mask = ((constructor_results['year'] == year) & (constructor_results['round'] < round_num))
                std_results = constructor_results[std_mask].copy()
                if len(std_results) > 0:
                    features['const_season_points'] = std_results['points'].sum()
                    features['const_season_avg_finish'] = str_results['finish_position'].mean()
                    features['const_season_wins'] = (str_results['position'] == 1).sum()
                    features['const_season_podiums'] = (str_results['position'] <= 3).sum()
                else:
                    features['const_season_points'] = 0
                    features['const_season_avg_finish'] = np.nan
                    features['const_season_wins'] = 0
                    features['const_season_podiums'] = 0

                constructor_features_list.append(features)

        constructor_features_df = pd.DataFrame(constructor_features_list)
        logger.info(f"Created constructor features for {len(constructor_features_df)} constructor-race combinations")
        return constructor_features_df

    def create_teammate_features(self,
                               driver_features_df: pd.DataFrame,
                               results_df: pd.DataFrame) -> pd.DataFrame:
        """Create teammate comparison features."""
        logger.info("Creating teammate features...")

        if driver_features_df.empty or results_df.empty:
            return driver_features_df

        # Get teammate information for each race
        teammate_data = []

        for _, race in results_df[['year', 'round', 'driver_id', 'constructor_id']].drop_duplicates().iterrows():
            year, round_num, constructor_id = race['year'], race['round'], race['constructor_id']

            # Get all drivers for this constructor in this race
            constructors_drivers = results_df[
                (results_df['year'] == year) &
                (results_df['round'] == round_num) &
                (results_df['constructor_id'] == constructor_id)
            ]['driver_id'].unique()

            # For each driver, calculate teammate comparison
            for driver_id in constructors_drivers:
                teammates = [d for d in constructors_drivers if d != driver_id]

                if len(teammates) > 0:
                    # Use first teammate (typically there's only one teammate per constructor)
                    teammate_id = teammates[0]

                    # Get historical performance comparison
                    driver_hist = results_df[
                        (results_df['driver_id'] == driver_id) &
                        ((results_df['year'] < year) |
                         ((results_df['year'] == year) & (results_df['round'] < round_num)))
                    ]
                    teammate_hist = results_df[
                        (results_df['driver_id'] == teammate_id) &
                        ((results_df['year'] < year) |
                         ((results_df['year'] == year) & (results_df['round'] < round_num)))
                    ]

                    # Calculate averages
                    driver_avg_finish = driver_hist['finish_position'].mean() if len(driver_hist) > 0 else np.nan
                    teammate_avg_finish = teammate_hist['finish_position'].mean() if len(teammate_hist) > 0 else np.nan
                    driver_avg_points = driver_hist['points'].mean() if len(driver_hist) > 0 else 0
                    teammate_avg_points = teammate_hist['points'].mean() if len(teammate_hist) > 0 else 0
                    driver_win_rate = (driver_hist['position'] == 1).sum() / len(driver_hist) if len(driver_hist) > 0 else 0
                    teammate_win_rate = (teammate_hist['position'] == 1).sum() / len(teammate_hist) if len(teammate_hist) > 0 else 0

                    teammate_data.append({
                        'year': year,
                        'round': round_num,
                        'driver_id': driver_id,
                        'teammate_id': teammate_id,
                        'teammate_avg_finish_diff': driver_avg_finish - teammate_avg_finish if not (pd.isna(driver_avg_finish) or pd.isna(teammate_avg_finish)) else np.nan,
                        'teammate_points_diff': driver_avg_points - teammate_avg_points,
                        'teammate_win_rate_diff': driver_win_rate - teammate_win_rate,
                        'teammate_experience_diff': len(driver_hist) - len(teammate_hist)
                    })

        if teammate_data:
            teammate_df = pd.DataFrame(teammate_data)
            # Merge with driver features
            enhanced_df = driver_features_df.merge(
                teammate_df,
                on=['year', 'round', 'driver_id'],
                how='left'
            )
            logger.info(f"Added teammate features for {len(teammate_df)} driver-race combinations")
            return enhanced_df
        else:
            logger.warning("No teammate data generated")
            return driver_features_df

    def create_circuit_features(self,
                              races_df: pd.DataFrame,
                              results_df: pd.DataFrame,
                              circuits_df: pd.DataFrame) -> pd.DataFrame:
        """Create circuit-level features."""
        logger.info("Creating circuit features...")

        if results_df.empty or circuits_df.empty:
            return pd.DataFrame()

        # Start with races and merge circuit info
        circuit_features_df = races_df[['year', 'round', 'circuit_id']].copy()
        circuit_features_df = circuit_features_df.merge(
            circuits_df[['circuit_id', 'circuit_name', 'location', 'country']],
            on='circuit_id',
            how='left'
        )

        circuit_features_list = []

        for _, race in circuit_features_df.iterrows():
            year, round_num, circuit_id = race['year'], race['round'], race['circuit_id']

            # Historical performance at this circuit
            hist_results = results_df[
                (results_df['circuit_id'] == circuit_id) &
                ((results_df['year'] < year) |
                 ((results_df['year'] == year) & (results_df['round'] < round_num)))
            ].copy()

            features = {
                'year': year,
                'round': round_num,
                'circuit_id': circuit_id,
                'circuit_name': race['circuit_name'],
                'location': race['location'],
                'country': race['country']
            }

            # Circuit-specific stats
            if len(hist_results) > 0:
                features['circuit_races_history'] = len(hist_results)
                features['circuit_avg_finish'] = hist_results['finish_position'].mean()
                features['circuit_win_rate'] = (hist_results['position'] == 1).sum() / len(hist_results)
                features['circuit_podium_rate'] = (hist_results['position'] <= 3).sum() / len(hist_results)
                features['circuit_dnf_rate'] = hist_results['dnf'].mean() if 'dnf' in hist_results.columns else 0
                features['circuit_avg_points'] = hist_results['points'].mean()

                # Constructor performance at this circuit
                if 'constructor_id' in hist_results.columns:
                    const_perf = hist_results.groupby('constructor_id').agg({
                        'points': 'mean',
                        'position': 'mean'
                    }).reset_index()
                    features['circuit_const_avg_points'] = const_perf['points'].mean() if len(const_perf) > 0 else 0
                    features['circuit_const_avg_position'] = const_perf['position'].mean() if len(const_perf) > 0 else np.nan
                else:
                    features['circuit_const_avg_points'] = 0
                    features['circuit_const_avg_position'] = np.nan
            else:
                features['circuit_races_history'] = 0
                features['circuit_avg_finish'] = np.nan
                features['circuit_win_rate'] = 0
                features['circuit_podium_rate'] = 0
                features['circuit_dnf_rate'] = 0
                features['circuit_avg_points'] = 0
                features['circuit_const_avg_points'] = 0
                features['circuit_const_avg_position'] = np.nan

            # Recent form at this circuit (last 3 races)
            recent_hist = hist_results.tail(3) if len(hist_results) >= 3 else hist_results
            if len(recent_hist) > 0:
                features['circuit_recent_3_avg_finish'] = recent_hist['finish_position'].mean()
                features['circuit_recent_3_dnf_rate'] = recent_hist['dnf'].mean() if 'dnf' in recent_hist.columns else 0
            else:
                features['circuit_recent_3_avg_finish'] = np.nan
                features['circuit_recent_3_dnf_rate'] = 0

            circuit_features_list.append(features)

        circuit_features_df = pd.DataFrame(circuit_features_list)
        logger.info(f"Created circuit features for {len(circuit_features_df)} circuit-race combinations")
        return circuit_features_df

    def create_championship_context_features(self,
                                           driver_features_df: pd.DataFrame,
                                           driver_standings_df: pd.DataFrame,
                                           constructor_standings_df: pd.DataFrame) -> pd.DataFrame:
        """Create championship context features."""
        logger.info("Creating championship context features...")

        if driver_features_df.empty:
            return driver_features_df

        # Start with driver features
        enhanced_df = driver_features_df.copy()

        # Add current championship position and points
        for idx, row in enhanced_df.iterrows():
            year, round_num, driver_id = row['year'], row['round'], row['driver_id']

            # Get driver standings before this race
            hist_standing = driver_standings_df[
                (driver_standings_df['driver_id'] == driver_id) &
                (driver_standings_df['year'] < year)
            ].copy()

            # If no historical standings, use most recent available
            if hist_standing.empty:
                hist_standing = driver_standings_df[
                    driver_standings_df['driver_id'] == driver_id
                ].copy()

            if not hist_standing.empty:
                # Get the most recent standing
                latest_standing = hist_standing.sort_values('year').iloc[-1]
                enhanced_df.at[idx, 'championship_position'] = latest_standing['position']
                enhanced_df.at[idx, 'championship_points'] = latest_standing['points']
                enhanced_df.at[idx, 'championship_wins'] = latest_standing['wins']
            else:
                enhanced_df.at[idx, 'championship_position'] = np.nan
                enhanced_df.at[idx, 'championship_points'] = 0
                enhanced_df.at[idx, 'championship_wins'] = 0

            # Get constructor standings
            const_id = row['constructor_id']
            if pd.notna(const_id):
                const_standing = constructor_standings_df[
                    (constructor_standings_df['constructor_id'] == const_id) &
                    (constructor_standings_df['year'] < year)
                ].copy()

                if const_standing.empty:
                    const_standing = constructor_standings_df[
                        (constructor_standings_df['constructor_id'] == const_id)
                    ].copy()

                if not const_standing.empty:
                    latest_const = const_standing.sort_values('year').iloc[-1]
                    enhanced_df.at[idx, 'constructor_championship_position'] = latest_const['position']
                    enhanced_df.at[idx, 'constructor_championship_points'] = latest_const['points']
                    enhanced_df.at[idx, 'constructor_wins'] = latest_const['wins']
                else:
                    enhanced_df.at[idx, 'constructor_championship_position'] = np.nan
                    enhanced_df.at[idx, 'constructor_championship_points'] = 0
                    enhanced_df.at[idx, 'constructor_wins'] = 0
            else:
                enhanced_df.at[idx, 'constructor_championship_position'] = np.nan
                enhanced_df.at[idx, 'constructor_championship_points'] = 0
                enhanced_df.at[idx, 'constructor_wins'] = 0

        # Calculate points gaps and championship context
        if not driver_standings_df.empty:
            # Get championship leader info for each race
            leader_info = []
            for _, row in enhanced_df.iterrows():
                year, round_num = row['year'], row['round']

                # Get standings after previous race (or before season start)
                prev_standings = driver_standings_df[
                    ((driver_standings_df['year'] < year) |
                     ((driver_standings_df['year'] == year) & (driver_standings_df['round'] < round_num)))
                ].copy()

                if not prev_standings.empty:
                    # Get leader after previous race
                    leader = prev_standings.loc[prev_standings['points'].idxmax()]
                    leader_info.append({
                        'year': year,
                        'round': round_num,
                        'championship_leader_points': leader['points'],
                        'championship_leader_id': leader['driver_id']
                    })
                else:
                    # At season start, no leader yet
                    leader_info.append({
                        'year': year,
                        'round': round_num,
                        'championship_leader_points': 0,
                        'championship_leader_id': None
                    })

            leader_df = pd.DataFrame(leader_info)
            enhanced_df = enhanced_df.merge(leader_df, on=['year', 'round'], how='left')

            # Calculate points gap to leader
            enhanced_df['points_gap_to_leader'] = enhanced_df['championship_leader_points'] - enhanced_df['championship_points']
            enhanced_df['points_gap_to_leader'] = enhanced_df['points_gap_to_leader'].fillna(0)

        # Races remaining in season
        if not races_df.empty:
            max_rounds_per_year = races_df.groupby('year')['round'].max().reset_index()
            max_rounds_per_year = max_rounds_per_year.rename(columns={'round': 'max_rounds'})
            enhanced_df = enhanced_df.merge(max_rounds_per_year, on='year', how='left')
            enhanced_df['races_remaining'] = enhanced_df['max_rounds'] - enhanced_df['round']
            enhanced_df['races_remaining'] = enhanced_df['races_remaining'].clip(lower=0)
            enhanced_df['season_progress'] = enhanced_df['round'] / enhanced_df['max_rounds']
            enhanced_df['season_progress'] = enhanced_df['season_progress'].fillna(0)
            enhanced_df = enhanced_df.drop('max_rounds', axis=1)
        else:
            enhanced_df['races_remaining'] = 0
            enhanced_df['season_progress'] = 0

        logger.info(f"Added championship context features to {len(enhanced_df)} driver-race combinations")
        return enhanced_df

    def engineer_all_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Engineer all features and return unified feature DataFrame."""
        logger.info("Starting comprehensive feature engineering...")

        # Extract dataframes
        races_df = data_dict.get('races', pd.DataFrame())
        results_df = data_dict.get('results', pd.DataFrame())
        qualifying_df = data_dict.get('qualifying', pd.DataFrame())
        driver_standings_df = data_dict.get('driver_standings', pd.DataFrame())
        constructor_standings_df = data_dict.get('constructor_standings', pd.DataFrame())
        circuits_df = data_dict.get('circuits', pd.DataFrame())

        # Create base driver features
        driver_features = self.create_driver_features(
            races_df, results_df, qualifying_df, driver_standings_df
        )

        # Add constructor features
        constructor_features = self.create_constructor_features(
            races_df, results_df, constructor_standings_df
        )

        # Merge constructor features
        if not constructor_features.empty and not driver_features.empty:
            driver_features = driver_features.merge(
                constructor_features,
                on=['year', 'round', 'constructor_id'],
                how='left',
                suffixes=('', '_const')
            )

        # Add teammate features
        driver_features = self.create_teammate_features(driver_features, results_df)

        # Add circuit features
        circuit_features = self.create_circuit_features(races_df, results_df, circuits_df)
        if not circuit_features.empty and not driver_features.empty:
            driver_features = driver_features.merge(
                circuit_features,
                on=['year', 'round', 'circuit_id'],
                how='left'
            )

        # Add championship context features
        driver_features = self.create_championship_context_features(
            driver_features, driver_standings_df, constructor_standings_df
        )

        logger.info(f"Feature engineering complete. Final feature set has {len(driver_features)} rows and {len(driver_features.columns)} columns")
        return driver_features

def main():
    """Main function for testing feature engineering."""
    # This would typically be called from the main pipeline
    print("Feature engineering module ready")

if __name__ == "__main__":
    main()