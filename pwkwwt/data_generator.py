import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import argparse


class DischargeDataGenerator:
    """
    Generate synthetic data for discharge impact model testing
    """

    def __init__(self,
                 start_date="2023-01-01",
                 end_date="2023-12-31",
                 num_downstream_points=3,
                 output_dir="test_data"):
        """
        Initialize the data generator

        Args:
            start_date: Start date for the data (string in format "YYYY-MM-DD")
            end_date: End date for the data (string in format "YYYY-MM-DD")
            num_downstream_points: Number of downstream monitoring points
            output_dir: Directory to save the generated data
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.num_days = (self.end_date - self.start_date).days + 1
        self.dates = [self.start_date + timedelta(days=i) for i in range(self.num_days)]
        self.num_downstream_points = num_downstream_points
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set random seed for reproducibility
        np.random.seed(42)

    def generate_discharge_data(self,
                                pollutants=None,
                                seasonal_factors=True,
                                weekly_patterns=True):
        """
        Generate synthetic discharge data

        Args:
            pollutants: List of pollutants to include (default: COD, NH3-N, TP)
            seasonal_factors: Whether to include seasonal variations
            weekly_patterns: Whether to include weekly patterns

        Returns:
            DataFrame with synthetic discharge data
        """
        if pollutants is None:
            pollutants = ['COD', 'NH3-N', 'TP']

        # Create empty dataframe
        df = pd.DataFrame()
        df['Date'] = self.dates

        # Flow rate (cubic meters per day)
        base_flow = 5000  # Base flow rate
        flow_variation = 2000  # Variation in flow rate

        # Generate flow with seasonal and random variations
        flow = np.zeros(self.num_days)
        for i in range(self.num_days):
            # Base flow
            day_flow = base_flow

            # Add seasonal variation
            if seasonal_factors:
                day_of_year = self.dates[i].timetuple().tm_yday
                seasonal_factor = np.sin(2 * np.pi * day_of_year / 365)
                day_flow += flow_variation * seasonal_factor

            # Add weekly variation (less on weekends)
            if weekly_patterns:
                weekday = self.dates[i].weekday()
                if weekday >= 5:  # Weekend
                    day_flow *= 0.7

            # Add random noise
            day_flow += np.random.normal(0, flow_variation * 0.1)

            # Ensure minimum flow
            flow[i] = max(day_flow, base_flow * 0.5)

        df['Flow'] = flow

        # Generate pollutant concentrations
        for pollutant in pollutants:
            # Set base values and variations for each pollutant
            if pollutant == 'COD':
                base_value = 200  # mg/L
                variation = 50
                seasonal_amplitude = 30
            elif pollutant == 'NH3-N':
                base_value = 25  # mg/L
                variation = 10
                seasonal_amplitude = 5
            elif pollutant == 'TP':
                base_value = 3  # mg/L
                variation = 1
                seasonal_amplitude = 0.8
            else:
                base_value = 10  # Generic value for other pollutants
                variation = 3
                seasonal_amplitude = 2

            # Generate concentration with seasonal and random variations
            concentration = np.zeros(self.num_days)
            for i in range(self.num_days):
                # Base concentration
                day_conc = base_value

                # Add seasonal variation
                if seasonal_factors:
                    day_of_year = self.dates[i].timetuple().tm_yday
                    # Phase shift for different pollutants
                    phase_shift = 0 if pollutant == 'COD' else np.pi / 2 if pollutant == 'NH3-N' else np.pi
                    seasonal_factor = np.sin(2 * np.pi * day_of_year / 365 + phase_shift)
                    day_conc += seasonal_amplitude * seasonal_factor

                # Add weekly variation (more on workdays due to industrial activity)
                if weekly_patterns:
                    weekday = self.dates[i].weekday()
                    if weekday < 5:  # Weekday
                        day_conc *= 1.2

                # Add random noise
                day_conc += np.random.normal(0, variation * 0.15)

                # Ensure non-negative
                concentration[i] = max(day_conc, 0)

            df[pollutant] = concentration

        # Calculate loads
        for pollutant in pollutants:
            df[f'{pollutant}_Load'] = df['Flow'] * df[pollutant] / 1000  # kg/day

        # Save to CSV
        output_path = os.path.join(self.output_dir, 'discharge_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Discharge data saved to {output_path}")

        # Generate a plot for visualization
        self._plot_discharge_data(df, pollutants)

        return df

    def generate_downstream_data(self,
                                 discharge_df,
                                 pollutants=None,
                                 distance_factors=None,
                                 transport_delay=None,
                                 degradation_rates=None):
        """
        Generate synthetic downstream monitoring data based on discharge data

        Args:
            discharge_df: DataFrame with discharge data
            pollutants: List of pollutants to include (default: COD, NH3-N, TP)
            distance_factors: Dictionary with distance factors for each point
            transport_delay: Dictionary with transport delay days for each point
            degradation_rates: Dictionary with degradation rates for each pollutant

        Returns:
            DataFrame with synthetic downstream data
        """
        if pollutants is None:
            pollutants = ['COD', 'NH3-N', 'TP']

        # Default distance factors (affects dilution)
        if distance_factors is None:
            distance_factors = {f'DS{i + 1}': 1.0 + i * 0.5 for i in range(self.num_downstream_points)}

        # Default transport delay in days
        if transport_delay is None:
            transport_delay = {f'DS{i + 1}': i + 1 for i in range(self.num_downstream_points)}

        # Default degradation rates (per day)
        if degradation_rates is None:
            degradation_rates = {
                'COD': 0.2,  # Faster degradation
                'NH3-N': 0.1,  # Moderate degradation
                'TP': 0.05  # Slower degradation
            }

        # Create empty dataframe
        all_data = []

        # Base background levels for each pollutant
        background_levels = {
            'COD': 10,  # mg/L
            'NH3-N': 0.5,  # mg/L
            'TP': 0.1  # mg/L
        }

        # For each monitoring point
        for point_id in range(1, self.num_downstream_points + 1):
            point_name = f'DS{point_id}'

            # Create point dataframe
            df = pd.DataFrame()
            df['Date'] = self.dates
            df['PointID'] = point_name

            # Distance factor for this point (affects dilution)
            distance = distance_factors[point_name]

            # Transport delay from discharge to this point
            delay = transport_delay[point_name]

            # Generate water quality parameters
            for pollutant in pollutants:
                # Background level for this pollutant
                background = background_levels.get(pollutant, 1.0)

                # Degradation rate for this pollutant
                degradation = degradation_rates.get(pollutant, 0.1)

                # Create array for pollutant concentration
                concentration = np.zeros(self.num_days)

                for i in range(self.num_days):
                    # Start with background level
                    day_conc = background

                    # Add impact from discharge (with delay)
                    discharge_idx = i - delay
                    if discharge_idx >= 0:
                        # Calculate impact based on discharge concentration, flow, and distance
                        discharge_conc = discharge_df.iloc[discharge_idx][pollutant]
                        discharge_flow = discharge_df.iloc[discharge_idx]['Flow']

                        # Simple model: impact decreases with distance and degrades over time
                        impact = discharge_conc / (distance * distance)
                        impact *= np.exp(-degradation * delay)

                        # Scale impact by flow (higher flow = more impact)
                        impact *= discharge_flow / 5000  # Normalized by typical flow

                        day_conc += impact

                    # Add random variations (e.g., from weather, other sources)
                    day_conc += np.random.normal(0, background * 0.2)

                    # Ensure non-negative
                    concentration[i] = max(day_conc, 0)

                df[pollutant] = concentration

            # Add other water quality parameters if needed
            df['Temperature'] = 20 + 10 * np.sin(2 * np.pi * np.arange(self.num_days) / 365) + np.random.normal(0, 1,
                                                                                                                self.num_days)
            df['pH'] = 7.0 + np.random.normal(0, 0.3, self.num_days)
            df['DO'] = 8.0 - 0.02 * df['Temperature'] + np.random.normal(0, 0.5, self.num_days)

            # Append to all data
            all_data.append(df)

        # Combine all points into one dataframe
        combined_df = pd.concat(all_data, ignore_index=True)

        # Save to CSV
        output_path = os.path.join(self.output_dir, 'downstream_data.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"Downstream data saved to {output_path}")

        # Generate a plot for visualization
        self._plot_downstream_data(combined_df, pollutants)

        return combined_df

    def generate_upstream_data(self,
                               pollutants=None,
                               seasonal_factors=True):
        """
        Generate synthetic upstream water quality data

        Args:
            pollutants: List of pollutants to include (default: COD, NH3-N, TP)
            seasonal_factors: Whether to include seasonal variations

        Returns:
            DataFrame with synthetic upstream data
        """
        if pollutants is None:
            pollutants = ['COD', 'NH3-N', 'TP']

        # Create empty dataframe
        df = pd.DataFrame()
        df['Date'] = self.dates

        # Base background levels for each pollutant
        background_levels = {
            'COD': 8,  # mg/L
            'NH3-N': 0.3,  # mg/L
            'TP': 0.08  # mg/L
        }

        # Generate water quality parameters
        for pollutant in pollutants:
            # Background level for this pollutant
            background = background_levels.get(pollutant, 0.5)

            # Create array for pollutant concentration
            concentration = np.zeros(self.num_days)

            for i in range(self.num_days):
                # Start with background level
                day_conc = background

                # Add seasonal variation
                if seasonal_factors:
                    day_of_year = self.dates[i].timetuple().tm_yday
                    seasonal_factor = np.sin(2 * np.pi * day_of_year / 365)
                    day_conc += background * 0.3 * seasonal_factor

                # Add random variations
                day_conc += np.random.normal(0, background * 0.15)

                # Ensure non-negative
                concentration[i] = max(day_conc, 0)

            df[pollutant] = concentration

        # Add other water quality parameters
        df['Temperature'] = 18 + 10 * np.sin(2 * np.pi * np.arange(self.num_days) / 365) + np.random.normal(0, 0.8,
                                                                                                            self.num_days)
        df['pH'] = 7.2 + np.random.normal(0, 0.2, self.num_days)
        df['DO'] = 8.5 - 0.02 * df['Temperature'] + np.random.normal(0, 0.4, self.num_days)

        # Save to CSV
        output_path = os.path.join(self.output_dir, 'upstream_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Upstream data saved to {output_path}")

        return df

    def generate_environment_data(self):
        """
        Generate synthetic environmental data (rainfall, temperature, etc.)

        Returns:
            DataFrame with synthetic environmental data
        """
        # Create empty dataframe
        df = pd.DataFrame()
        df['Date'] = self.dates

        # Temperature (°C) with seasonal variation
        temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(self.num_days) / 365)
        temperature += np.random.normal(0, 2, self.num_days)  # Add random variations
        df['Temperature'] = temperature

        # Rainfall (mm) with seasonal variation and random events
        rainfall_base = 2 + 3 * np.sin(
            2 * np.pi * np.arange(self.num_days) / 365 + np.pi)  # Phase shifted from temperature
        rainfall = np.zeros(self.num_days)

        # Add seasonal pattern and random rain events
        for i in range(self.num_days):
            # Base rainfall for this day
            day_rain = max(0, rainfall_base[i] + np.random.normal(0, 1))

            # Randomly add heavy rainfall events (more likely in rainy season)
            if rainfall_base[i] > 3:  # Rainy season
                if np.random.random() < 0.1:  # 10% chance of heavy rain
                    day_rain += np.random.gamma(2, 10)  # Heavy rain event
            else:  # Dry season
                if np.random.random() < 0.03:  # 3% chance of heavy rain
                    day_rain += np.random.gamma(1, 8)  # Less intense

            rainfall[i] = day_rain

        df['Rainfall'] = rainfall

        # Wind speed (m/s)
        df['WindSpeed'] = 3 + np.random.gamma(1, 1, self.num_days)

        # Relative humidity (%)
        humidity_base = 65 + 15 * np.sin(2 * np.pi * np.arange(self.num_days) / 365)
        df['Humidity'] = humidity_base + np.random.normal(0, 5, self.num_days)
        df['Humidity'] = df['Humidity'].clip(30, 100)  # Limit to realistic range

        # Solar radiation (W/m²) - correlated with inverse of rainfall and seasonal
        radiation_base = 200 + 150 * np.sin(2 * np.pi * np.arange(self.num_days) / 365)
        radiation = radiation_base * (1 - 0.3 * rainfall / rainfall.max())  # Less radiation on rainy days
        df['SolarRadiation'] = radiation + np.random.normal(0, 20, self.num_days)
        df['SolarRadiation'] = df['SolarRadiation'].clip(0, 1000)  # Limit to realistic range

        # Save to CSV
        output_path = os.path.join(self.output_dir, 'environment_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Environment data saved to {output_path}")

        # Generate a plot for visualization
        self._plot_environment_data(df)

        return df

    def _plot_discharge_data(self, df, pollutants):
        """Create plots for discharge data"""
        plt.figure(figsize=(15, 10))

        # Plot Flow
        plt.subplot(len(pollutants) + 1, 1, 1)
        plt.plot(df['Date'], df['Flow'], 'b-')
        plt.title('Discharge Flow Rate')
        plt.ylabel('Flow (m³/day)')
        plt.grid(True, alpha=0.3)

        # Plot pollutants
        for i, pollutant in enumerate(pollutants):
            plt.subplot(len(pollutants) + 1, 1, i + 2)
            plt.plot(df['Date'], df[pollutant], 'r-')
            plt.title(f'Discharge {pollutant} Concentration')
            plt.ylabel(f'{pollutant} (mg/L)')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'discharge_data_plot.png'), dpi=300)
        plt.close()

    def _plot_downstream_data(self, df, pollutants):
        """Create plots for downstream data"""
        plt.figure(figsize=(15, 12))

        # Get unique points
        points = df['PointID'].unique()

        # For each pollutant
        for i, pollutant in enumerate(pollutants):
            plt.subplot(len(pollutants), 1, i + 1)

            # Plot each monitoring point
            for point in points:
                point_data = df[df['PointID'] == point]
                plt.plot(point_data['Date'], point_data[pollutant], label=point)

            plt.title(f'Downstream {pollutant} Concentration')
            plt.ylabel(f'{pollutant} (mg/L)')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'downstream_data_plot.png'), dpi=300)
        plt.close()

    def _plot_environment_data(self, df):
        """Create plots for environment data"""
        plt.figure(figsize=(15, 12))

        # Plot temperature
        plt.subplot(4, 1, 1)
        plt.plot(df['Date'], df['Temperature'], 'r-')
        plt.title('Temperature')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, alpha=0.3)

        # Plot rainfall
        plt.subplot(4, 1, 2)
        plt.bar(df['Date'], df['Rainfall'], color='b', width=1.0)
        plt.title('Rainfall')
        plt.ylabel('Rainfall (mm)')
        plt.grid(True, alpha=0.3)

        # Plot humidity
        plt.subplot(4, 1, 3)
        plt.plot(df['Date'], df['Humidity'], 'g-')
        plt.title('Relative Humidity')
        plt.ylabel('Humidity (%)')
        plt.grid(True, alpha=0.3)

        # Plot solar radiation
        plt.subplot(4, 1, 4)
        plt.plot(df['Date'], df['SolarRadiation'], 'orange')
        plt.title('Solar Radiation')
        plt.ylabel('Radiation (W/m²)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'environment_data_plot.png'), dpi=300)
        plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic data for discharge impact model')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--num_points', type=int, default=3, help='Number of downstream monitoring points')
    parser.add_argument('--output_dir', type=str, default='test_data', help='Output directory')
    args = parser.parse_args()

    # Create data generator
    generator = DischargeDataGenerator(
        start_date=args.start_date,
        end_date=args.end_date,
        num_downstream_points=args.num_points,
        output_dir=args.output_dir
    )

    # Generate data
    discharge_df = generator.generate_discharge_data()
    generator.generate_downstream_data(discharge_df)
    generator.generate_upstream_data()
    generator.generate_environment_data()

    print("\nData generation complete. Files saved to:", args.output_dir)


if __name__ == "__main__":
    main()