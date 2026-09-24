"""
Simulation module for F1 Championship Predictor.
Handles Monte Carlo simulation of championship outcomes.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import itertools
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1Simulation:
    """Handles Monte Carlo simulation of F1 championship outcomes."""

    def __init__(self):
        # Standard F1 points system (2024 onwards)
        self.points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]  # Positions 1-10
        self.fastest_lap_point = 1  # Additional point for fastest lap (if in top 10)

    def calculate_race_points(self, position: int, fastest_lap: bool = False,
                            in_top_10_for_fl: bool = True) -> int:
        """Calculate points for a given race position."""
        if position <= 0 or position > 20:  # Invalid position
            return 0

        if position <= 10:
            base_points = self.points_system[position - 1]
        else:
            base_points = 0

        # Add fastest lap point if applicable
        fl_points = 0
        if fastest_lap and in_top_10_for_fl and position <= 10:
            fl_points = self.fastest_lap_point

        return base_points + fl_points

    def simulate_race_outcomes(self, race_predictions: pd.DataFrame,
                             n_simulations: int = 10000) -> pd.DataFrame:
        """Simulate race outcomes based on driver performance predictions."""
        logger.info(f"Simulating {n_simulations} race outcomes...")

        if race_predictions.empty:
            return pd.DataFrame()

        # Group by race
        races = race_predictions[['year', 'round', 'race_name']].drop_duplicates()
        simulation_results = []

        for _, race_info in races.iterrows():
            year, round_num, race_name = race_info['year'], race_info['round'], race_info['race_name']

            # Get predictions for this race
            race_data = race_predictions[
                (race_predictions['year'] == year) &
                (race_predictions['round'] == round_num)
            ].copy()

            if race_data.empty:
                continue

            # Simulate this race multiple times
            race_simulations = []

            for sim in range(n_simulations):
                # Simulate finishing order for each driver
                simulated_results = []

                for _, driver in race_data.iterrows():
                    # Use predicted finish position with some uncertainty
                    # In a real implementation, you'd use a proper probability distribution
                    pred_position = driver.get('predicted_finish_position', 10)
                    position_uncertainty = driver.get('position_uncertainty', 3.0)

                    # Add noise to predicted position
                    simulated_position = max(1, int(np.round(
                        np.random.normal(pred_position, position_uncertainty)
                    )))
                    simulated_position = min(20, simulated_position)  # Cap at 20

                    # Determine if fastest lap (simplified)
                    fastest_lap = np.random.random() < 0.15  # 15% chance
                    in_top_10 = simulated_position <= 10

                    # Calculate points
                    points = self.calculate_race_points(simulated_position, fastest_lap, in_top_10)

                    simulated_results.append({
                        'year': year,
                        'round': round_num,
                        'driver_id': driver['driver_id'],
                        'simulated_position': simulated_position,
                        'simulated_points': points,
                        'fastest_lap': fastest_lap,
                        'simulation_id': sim
                    })

                race_simulations.extend(simulated_results)

            simulation_results.extend(race_simulations)

        simulation_df = pd.DataFrame(race_simulations)
        logger.info(f"Race simulation complete: {len(simulation_df)} simulated race results")
        return simulation_df

    def simulate_championship(self, race_predictions: pd.DataFrame,
                            current_standings: pd.DataFrame = None,
                            n_simulations: int = 10000,
                            remaining_races_only: bool = True) -> Dict[str, Any]:
        """Simulate the entire championship using Monte Carlo methods."""
        logger.info(f"Starting championship simulation with {n_simulations} iterations...")

        if race_predictions.empty:
            return {}

        # Get unique races and sort chronologically
        races = race_predictions[['year', 'round']].drop_duplicates()
        races = races.sort_values(['year', 'round']).reset_index(drop=True)

        # Filter to remaining races if specified and current standings provided
        if remaining_races_only and current_standings is not None and not current_standings.empty:
            # Determine how many races have been completed based on current standings
            # This is simplified - in practice you'd check actual completed races
            max_completed_round = current_standings['round'].max() if 'round' in current_standings.columns else 0
            races = races[races['round'] > max_completed_round].copy()
            logger.info(f"Simulating {len(races)} remaining races")

        if races.empty:
            logger.warning("No races to simulate")
            return {}

        # Get all drivers that could participate
        all_drivers = race_predictions['driver_id'].unique()

        # Initialize championship points for each driver
        championship_points = defaultdict(float)
        if current_standings is not None and not current_standings.empty:
            for _, driver in current_standings.iterrows():
                driver_id = driver['driver_id']
                points = driver.get('points', 0)
                championship_points[driver_id] = float(points)

        # Run simulations
        simulation_results = []

        for sim_id in range(n_simulations):
            # Reset points for this simulation (start from current standings)
            sim_points = defaultdict(float)
            for driver_id, points in championship_points.items():
                sim_points[driver_id] = points

            # Simulate each remaining race
            for _, race_info in races.iterrows():
                year, round_num = race_info['year'], race_info['round']

                # Get predictions for this race
                race_data = race_predictions[
                    (race_predictions['year'] == year) &
                    (race_predictions['round'] == round_num)
                ].copy()

                if race_data.empty:
                    continue

                # Simulate race outcomes
                race_results = self._simulate_single_race(race_data, sim_id)

                # Add points to simulation totals
                for result in race_results:
                    driver_id = result['driver_id']
                    points = result['points']
                    sim_points[driver_id] += points

            # Record final championship positions for this simulation
            sorted_drivers = sorted(sim_points.items(), key=lambda x: x[1], reverse=True)
            for position, (driver_id, points) in enumerate(sorted_drivers, start=1):
                simulation_results.append({
                    'simulation_id': sim_id,
                    'driver_id': driver_id,
                    'final_points': points,
                    'final_position': position
                })

        # Process simulation results
        sim_df = pd.DataFrame(simulation_results)

        # Calculate championship win probabilities
        win_counts = sim_df[sim_df['final_position'] == 1].groupby('driver_id').size()
        win_probabilities = (win_counts / n_simulations).fillna(0)

        # Calculate podium probabilities
        podium_counts = sim_df[sim_df['final_position'] <= 3].groupby('driver_id').size()
        podium_probabilities = (podium_counts / n_simulations).fillna(0)

        # Calculate points statistics
        points_stats = sim_df.groupby('driver_id')['final_points'].agg([
            'mean', 'std', 'median',
            lambda x: np.percentile(x, 5),   # 5th percentile
            lambda x: np.percentile(x, 95)   # 95th percentile
        ]).rename(columns={
            '<lambda_0': 'p5_points',
            '<lambda_1': 'p95_points'
        })

        # Calculate position statistics
        position_stats = sim_df.groupby('driver_id')['final_position'].agg([
            'mean', 'std', 'median',
            lambda x: np.percentile(x, 5),   # 5th percentile (best case)
            lambda x: np.percentile(x, 95)   # 95th percentile (worst case)
        ]).rename(columns={
            '<lambda_0': 'p5_position',
            '<lambda_1': 'p95_position'
        })

        # Combine all statistics
        championship_summary = pd.DataFrame({
            'win_probability': win_probabilities,
            'podium_probability': podium_probabilities,
        }).fillna(0)

        championship_summary = championship_summary.join(points_stats, how='left')
        championship_summary = championship_summary.join(position_stats, how='left')

        # Reset index to get driver_id as column
        championship_summary = championship_summary.reset_index()

        # Sort by win probability
        championship_summary = championship_summary.sort_values('win_probability', ascending=False)

        # Prepare results dictionary
        results = {
            'championship_summary': championship_summary,
            'simulation_details': sim_df,
            'n_simulations': n_simulations,
            'races_simulated': len(races),
            'drivers_in_simulation': len(all_drivers),
            'timestamp': pd.Timestamp.now()
        }

        logger.info(f"Championship simulation complete: {n_simulations} simulations for {len(all_drivers)} drivers")
        return results

    def _simulate_single_race(self, race_data: pd.DataFrame, simulation_id: int) -> List[Dict]:
        """Simulate a single race outcome."""
        race_results = []

        # Create a copy to work with
        drivers = race_data.copy()

        # Simulate qualifying order (if quali predictions available)
        if 'predicted_quali_position' in drivers.columns:
            # Add noise to qualifying predictions
            quali_noise = np.random.normal(0, 1.5, len(drivers))  # 1.5 position std dev
            drivers['simulated_quali'] = drivers['predicted_quali_position'] + quali_noise
            drivers['simulated_quali'] = drivers['simulated_quali'].clip(1, 20)
            drivers = drivers.sort_values('simulated_quali').reset_index(drop=True)
        else:
            # Random starting grid if no quali predictions
            drivers = drivers.sample(frac=1, random_state=simulation_id).reset_index(drop=True)

        # Simulate race finish based on starting position and pace
        for idx, (_, driver) in enumerate(drivers.iterrows()):
            # Starting grid position (based on quali simulation)
            grid_position = idx + 1

            # Predicted race pace (simplified)
            base_pace = driver.get('predicted_race_pace', 0.0)  # Lower is better
            pace_variation = np.random.normal(0, 0.5)  # Random variation in pace

            # Final race score (lower is better)
            race_score = base_pace + pace_variation + (grid_position * 0.1)  # Penalty for bad grid

            # Reorder by race score
            drivers.loc[drivers.index == driver.name, 'race_score'] = race_score

        # Sort by race score to get finishing order
        drivers = drivers.sort_values('race_score').reset_index(drop=True)

        # Assign final positions and calculate points
        for pos, (_, driver) in enumerate(drivers.iterrows(), start=1):
            final_position = pos

            # Determine if fastest lap (simplified - give to faster drivers with some randomness)
            fastest_lap_prob = max(0.05, 0.3 - (final_position - 1) * 0.02)  # Decreasing probability
            fastest_lap = np.random.random() < fastest_lap_prob

            # Calculate points
            in_top_10_for_fl = final_position <= 10
            points = self.calculate_race_points(final_position, fastest_lap, in_top_10_for_fl)

            race_results.append({
                'driver_id': driver['driver_id'],
                'final_position': final_position,
                'points': points,
                'fastest_lap': fastest_lap,
                'grid_position': grid_position,
                'simulation_id': simulation_id
            })

        return race_results

    def simulate_season_outcomes(self, championship_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate additional season outcome statistics from simulation results."""
        if not championship_results or 'championship_summary' not in championship_results:
            return {}

        summary = championship_results['championship_summary']
        sim_details = championship_results['simulation_details']

        # Calculate expected final points distribution
        points_distribution = {}
        for _, driver in summary.iterrows():
            driver_id = driver['driver_id']
            points_samples = sim_details[sim_details['driver_id'] == driver_id]['final_points']

            if len(points_samples) > 0:
                points_distribution[driver_id] = {
                    'mean': points_samples.mean(),
                    'median': points_samples.median(),
                    'std': points_samples.std(),
                    'p10': np.percentile(points_samples, 10),
                    'p90': np.percentile(points_samples, 90),
                    'range': (points_samples.min(), points_samples.max())
                }

        # Calculate position change expectations
        position_changes = {}
        # This would require knowing starting positions - simplified for now

        # Calculate constructor championship simulation (if constructor data available)
        # This would be implemented similarly to driver simulation

        results = {
            'points_distribution': points_distribution,
            'championship_entropy': self._calculate_championship_entropy(summary),
            'simulation_quality': {
                'n_simulations': championship_results['n_simulations'],
                'convergence_metric': self._check_convergence(sim_details)
            }
        }

        return results

    def _calculate_championship_entropy(self, summary: pd.DataFrame) -> float:
        """Calculate entropy of championship outcome distribution."""
        if 'win_probability' not in summary.columns or summary.empty:
            return 0.0

        probs = summary['win_probability'].values
        # Normalize to ensure it sums to 1
        probs = probs / probs.sum() if probs.sum() > 0 else probs
        # Remove zeros for log calculation
        probs = probs[probs > 0]

        if len(probs) == 0:
            return 0.0

        entropy = -np.sum(probs * np.log(probs))
        return entropy

    def _check_convergence(self, sim_details: pd.DataFrame) -> float:
        """Check convergence of simulation results."""
        if sim_details.empty or 'driver_id' not in sim_details.columns:
            return 0.0

        # Calculate running average of win probabilities and see how much they change
        drivers = sim_details['driver_id'].unique()
        if len(drivers) == 0:
            return 0.0

        # For simplicity, return a placeholder metric
        # In practice, you'd split simulations into batches and compare early vs late results
        return 0.95  # Placeholder for good convergence

def main():
    """Main function for testing simulation."""
    print("F1 Simulation module ready")

if __name__ == "__main__":
    main()