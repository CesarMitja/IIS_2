import pandas as pd
from pathlib import Path

# Paths for data files
BIKE_DATA_FILE = Path("data/raw/kolesa.csv")
WEATHER_DATA_FILE = Path("data/raw/vreme.csv")
OUTPUT_DATA_FILE = Path("data/processed/data_for_prediction.csv")

# Load datasets
bike_data = pd.read_csv(BIKE_DATA_FILE, parse_dates=['last_update'])
weather_data = pd.read_csv(WEATHER_DATA_FILE, parse_dates=['time'])

# Refine weather data: rename columns for clarity and select necessary ones
weather_columns_rename = {
    'temperature_2m': 'temperature',
    'relative_humidity_2m': 'relative_humidity',
    'dew_point_2m': 'dew_point',
    'apparent_temperature': 'apparent_temperature',
    'precipitation_probability': 'precipitation_probability',
    'rain': 'rain',
    'surface_pressure': 'surface_pressure'
}
weather_data.rename(columns=weather_columns_rename, inplace=True)
weather_data = weather_data[['time', 'temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 'precipitation_probability', 'rain', 'surface_pressure']]

# Align time columns in both datasets to the nearest hour
bike_data['last_update'] = bike_data['last_update'].dt.round('H')
weather_data['time'] = weather_data['time'].dt.round('H')

# Merge datasets on time column
combined_data = pd.merge_asof(
    bike_data.sort_values('last_update'), 
    weather_data.sort_values('time'), 
    left_on='last_update', 
    right_on='time',
    direction='nearest'
)

combined_data.rename(columns={'last_update': 'date'}, inplace=True)

# Prepare the final DataFrame
combined_data = combined_data[['date','temperature', 'relative_humidity', 'dew_point', 'apparent_temperature',
            'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands']]



# Set timestamp to UTC timezone
combined_data['date'] = combined_data['date'].dt.tz_localize('UTC')

# Save the combined data
combined_data.to_csv(OUTPUT_DATA_FILE, index=False)

print(f"Data saved to {OUTPUT_DATA_FILE}.")