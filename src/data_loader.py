"""
Data loader for F1 Championship Predictor.
Handles downloading data from Jolpica F1 API (Ergast-compatible).
"""

import requests
import pandas as pd
import json
import os
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1DataLoader:
    """Handles downloading and caching F1 data from Jolpica API."""

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = cache_dir
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a request to the Ergast API with caching."""
        if params is None:
            params = {}

        # Create cache filename
        cache_key = f"{endpoint}_{hash(str(sorted(params.items())))}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        # Check if cached data exists and is recent (less than 24 hours old)
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 24 * 3600:  # 24 hours
                logger.info(f"Loading cached data for {endpoint}")
                with open(cache_file, 'r') as f:
                    return json.load(f)

        # Make API request to Ergast (no API key needed)
        logger.info(f"Fetching data from API: {endpoint}")
        ergast_url = f"http://ergast.com/api/f1/{endpoint.lstrip('/')}.json"

        try:
            response = requests.get(ergast_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Cache the response
            with open(cache_file, 'w') as f:
                json.dump(data, f)

            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from {endpoint}: {e}")
            # Try to load old cache if available
            if os.path.exists(cache_file):
                logger.warning(f"Using stale cache for {endpoint}")
                with open(cache_file, 'r') as f:
                    return json.load(f)
            raise

    def get_seasons(self, start_year: int = 2015, end_year: Optional[int] = None) -> pd.DataFrame:
        """Get list of F1 seasons."""
        if end_year is None:
            end_year = datetime.now().year

        data = self._make_request("", {"limit": 1000})  # Get all seasons
        seasons_data = data.get('MRData', {}).get('SeasonTable', {}).get('Seasons', [])

        # Filter by year range and convert to DataFrame
        seasons = []
        for season in seasons_data:
            year = int(season['year'])
            if start_year <= year <= end_year:
                seasons.append({
                    'year': year,
                    'url': season.get('url', '')
                })

        return pd.DataFrame(seasons)

    def get_races(self, year: int) -> pd.DataFrame:
        """Get all races for a given year."""
        data = self._make_request(f"{year}", {"limit": 1000})
        races_data = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])

        races = []
        for race in races_data:
            race_info = {
                'year': int(race['year']),
                'round': int(race['round']),
                'circuit_id': race['Circuit']['circuitId'],
                'circuit_name': race['Circuit']['circuitName'],
                'location': race['Circuit']['Location']['locality'],
                'country': race['Circuit']['Location']['country'],
                'lat': float(race['Circuit']['Location']['lat']),
                'lng': float(race['Circuit']['Location']['long']),
                'date': race['date'],
                'time': race.get('time', ''),
                'race_name': race['raceName'],
                'url': race['url']
            }

            # Handle sprint races if available
            if 'Sprint' in race:
                race_info['has_sprint'] = True
                race_info['sprint_date'] = race['Sprint']['date']
            else:
                race_info['has_sprint'] = False
                race_info['sprint_date'] = None

            races.append(race_info)

        return pd.DataFrame(races)

    def get_race_results(self, year: int, round: int) -> pd.DataFrame:
        """Get race results for a specific race."""
        data = self._make_request(f"{year}/{round}/results", {"limit": 1000})
        results_data = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])

        if not results_data:
            return pd.DataFrame()

        results = results_data[0].get('Results', [])

        race_results = []
        for result in results:
            driver_info = result['Driver']
            constructor_info = result['Constructor']

            # Handle time/status
            time_result = result.get('Time', {})
            status = result.get('status', '')

            race_result = {
                'year': int(race['year']),
                'round': int(race['round']),
                'driver_id': driver_info['driverId'],
                'driver_ref': driver_info.get('driverRef', ''),
                'driver_number': int(driver_info.get('permanentNumber', 0)) if driver_info.get('permanentNumber') else None,
                'driver_code': driver_info.get('code', ''),
                'driver_first_name': driver_info.get('givenName', ''),
                'driver_last_name': driver_info.get('familyName', ''),
                'constructor_id': constructor_info['constructorId'],
                'constructor_name': constructor_info['name'],
                'grid_position': int(result['grid']) if result['grid'].isdigit() else None,
                'finish_position': int(result['position']) if result['position'].isdigit() else None,
                'finish_position_text': result['position'],
                'points': float(result['points']),
                'laps': int(result['laps']) if result['laps'].isdigit() else None,
                'time': time_result.get('time', '') if time_result else '',
                'milliseconds': int(time_result.get('milliseconds', 0)) if time_result and time_result.get('milliseconds') else None,
                'fastest_lap': result.get('FastestLap', {}).get('rank', None),
                'fastest_lap_time': result.get('FastestLap', {}).get('Time', {}).get('time', '') if result.get('FastestLap') else '',
                'fastest_lap_speed': result.get('FastestLap', {}).get('AverageSpeed', {}).get('speed', '') if result.get('FastestLap') else '',
                'status': status
            }

            race_results.append(race_result)

        return pd.DataFrame(race_results)

    def get_qualifying_results(self, year: int, round: int) -> pd.DataFrame:
        """Get qualifying results for a specific race."""
        data = self._make_request(f"{year}/{round}/qualifying", {"limit": 1000})
        results_data = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])

        if not results_data:
            return pd.DataFrame()

        results = results_data[0].get('QualifyingResults', [])

        qualifying_results = []
        for result in results:
            driver_info = result['Driver']
            constructor_info = result['Constructor']

            quali_result = {
                'year': int(race['year']),
                'round': int(race['round']),
                'driver_id': driver_info['driverId'],
                'constructor_id': constructor_info['constructorId'],
                'position': int(result['position']) if result['position'].isdigit() else None,
                'q1': result.get('Q1', ''),
                'q2': result.get('Q2', ''),
                'q3': result.get('Q3', '')
            }

            qualifying_results.append(quali_result)

        return pd.DataFrame(qualifying_results)

    def get_driver_standings(self, year: int) -> pd.DataFrame:
        """Get driver standings for a given year."""
        data = self._make_request(f"{year}/driverStandings", {"limit": 1000})
        standings_data = data.get('MRData', {}).get('StandingsTable', {}).get('StandingsLists', [])

        if not standings_data:
            return pd.DataFrame()

        # Get the final standings (last race of the season)
        driver_standings = standings_data[-1].get('DriverStandings', [])

        standings = []
        for standing in driver_standings:
            driver_info = standing['Driver']
            constructor_info = standing['Constructors'][0] if standing['Constructors'] else {}

            standing_dict = {
                'year': int(standings_data[-1]['season']),
                'driver_id': driver_info['driverId'],
                'driver_ref': driver_info.get('driverRef', ''),
                'driver_number': int(driver_info.get('permanentNumber', 0)) if driver_info.get('permanentNumber') else None,
                'driver_code': driver_info.get('code', ''),
                'driver_first_name': driver_info.get('givenName', ''),
                'driver_last_name': driver_info.get('familyName', ''),
                'constructor_id': constructor_info.get('constructorId', ''),
                'constructor_name': constructor_info.get('name', ''),
                'points': float(standing['points']),
                'position': int(standing['position']),
                'wins': int(standing['wins'])
            }

            standings.append(standing_dict)

        return pd.DataFrame(standings)

    def get_constructor_standings(self, year: int) -> pd.DataFrame:
        """Get constructor standings for a given year."""
        data = self._make_request(f"{year}/constructorStandings", {"limit": 1000})
        standings_data = data.get('MRData', {}).get('StandingsTable', {}).get('StandingsLists', [])

        if not standings_data:
            return pd.DataFrame()

        # Get the final standings (last race of the season)
        constructor_standings = standings_data[-1].get('ConstructorStandings', [])

        standings = []
        for standing in constructor_standings:
            constructor_info = standing['Constructor']

            standing_dict = {
                'year': int(standings_data[-1]['season']),
                'constructor_id': constructor_info['constructorId'],
                'constructor_name': constructor_info['name'],
                'points': float(standing['points']),
                'position': int(standing['position']),
                'wins': int(standing['wins'])
            }

            standings.append(standing_dict)

        return pd.DataFrame(standings)

    def get_circuits(self) -> pd.DataFrame:
        """Get all circuits."""
        data = self._make_request("circuits", {"limit": 1000})
        circuits_data = data.get('MRData', {}).get('CircuitTable', {}).get('Circuits', [])

        circuits = []
        for circuit in circuits_data:
            circuit_dict = {
                'circuit_id': circuit['circuitId'],
                'circuit_name': circuit['circuitName'],
                'location': circuit['Location']['locality'],
                'country': circuit['Location']['country'],
                'lat': float(circuit['Location']['lat']),
                'lng': float(circuit['Location']['long']),
                'url': circuit['url']
            }

            circuits.append(circuit_dict)

        return pd.DataFrame(circuits)

    def download_historical_data(self, start_year: int = 2015, end_year: Optional[int] = None) -> None:
        """Download historical data for all seasons in range."""
        if end_year is None:
            end_year = datetime.now().year - 1  # Don't include current incomplete season by default

        logger.info(f"Downloading historical data from {start_year} to {end_year}")

        # Get seasons
        seasons_df = self.get_seasons(start_year, end_year)

        for _, season in seasons_df.iterrows():
            year = season['year']
            logger.info(f"Processing season {year}")

            # Get races for this season
            races_df = self.get_races(year)

            # Save races data
            races_df.to_parquet(f"{self.cache_dir}/races_{year}.parquet", index=False)

            # Process each race
            for _, race in races_df.iterrows():
                round_num = race['round']

                # Get race results
                results_df = self.get_race_results(year, round_num)
                if not results_df.empty:
                    results_df.to_parquet(
                        f"{self.cache_dir}/results_{year}_{round_num}.parquet",
                        index=False
                    )

                # Get qualifying results
                quali_df = self.get_qualifying_results(year, round_num)
                if not quali_df.empty:
                    quali_df.to_parquet(
                        f"{self.cache_dir}/qualifying_{year}_{round_num}.parquet",
                        index=False
                    )

                # Small delay to be respectful to the API
                time.sleep(0.1)

            # Get season standings
            driver_standings_df = self.get_driver_standings(year)
            if not driver_standings_df.empty:
                driver_standings_df.to_parquet(
                    f"{self.cache_dir}/driver_standings_{year}.parquet",
                    index=False
                )

            constructor_standings_df = self.get_constructor_standings(year)
            if not constructor_standings_df.empty:
                constructor_standings_df.to_parquet(
                    f"{self.cache_dir}/constructor_standings_{year}.parquet",
                    index=False
                )

            logger.info(f"Completed season {year}")

        # Save circuits data
        circuits_df = self.get_circuits()
        circuits_df.to_parquet(f"{self.cache_dir}/circuits.parquet", index=False)

        logger.info("Historical data download complete")

def main():
    """Main function for running the data loader from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Download F1 historical data")
    parser.add_argument("--init", action="store_true", help="Initialize data download")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year for data download")
    parser.add_argument("--end-year", type=int, default=None, help="End year for data download")

    args = parser.parse_args()

    if args.init:
        loader = F1DataLoader()
        loader.download_historical_data(start_year=args.start_year, end_year=args.end_year)
        print("Data initialization complete!")

if __name__ == "__main__":
    main()