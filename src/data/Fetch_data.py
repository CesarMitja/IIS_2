import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

API_URL = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
TARGET_STATION_NAME = "GOSPOSVETSKA C. - TURNERJEVA UL."
RAW_DATA_PATH = Path("data/raw")
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_PARAMS = {
    'latitude': 46.5547,
    'longitude': 15.6467,
    'hourly': 'temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,surface_pressure'
}

def fetch_weather():
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    next_hour = now + timedelta(hours=1)
    start_time = now.strftime('%Y-%m-%dT%H:00:00Z')
    end_time = next_hour.strftime('%Y-%m-%dT%H:00:00Z')

    WEATHER_PARAMS.update({'start': start_time, 'end': end_time})
    response = requests.get(WEATHER_API_URL, params=WEATHER_PARAMS)
    response.raise_for_status()
    return response.json()

def store_weather_data(weather_data):
    df = pd.DataFrame(weather_data['hourly'])
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    current_data = df[df['time'] == current_hour]

    if not current_data.empty:
        file_path = RAW_DATA_PATH / 'vreme.csv'
        current_data.to_csv(file_path, mode='a', header=not file_path.exists(), index=False)
    else:
        print("No current hour data available.")

def fetch_and_store_bike_data():
    response = requests.get(API_URL)
    response.raise_for_status()
    stations = response.json()
    for station in stations:
        if station['name'] == TARGET_STATION_NAME:
            df = pd.DataFrame([station])
            df['last_update'] = pd.to_datetime(df['last_update'], unit='ms')
            sorted_columns = ['last_update'] + [col for col in df.columns if col != 'last_update']
            df = df[sorted_columns]
            file_name = "kolesa.csv"
            file_path = RAW_DATA_PATH / file_name
            df.to_csv(file_path, mode='a', header=not file_path.exists(), index=False)

def main():
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    weather_data = fetch_weather()
    store_weather_data(weather_data)
    fetch_and_store_bike_data()
    print(f"Data saved to vreme and kolesa.csv.")

if __name__ == "__main__":
    main()