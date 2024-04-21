import requests
import pandas as pd

# URL za pridobivanje svežih podatkov
url = 'https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b'

# Pridobivanje podatkov
response = requests.get(url)
data = response.json()

# Pretvorba v pandas DataFrame
df = pd.DataFrame(data)

# Filtriranje za postajališče "GOSPOSVETSKA C. - TURNERJEVA UL."
filtered_df = df[df['name'] == "GOSPOSVETSKA C. - TURNERJEVA UL."]

# Obdelava podatkov
filtered_df['last_update'] = pd.to_datetime(filtered_df['last_update'], unit='ms')
filtered_df.set_index('last_update', inplace=True)

# Agregacija na urni interval
hourly_data = filtered_df.resample('H').mean()

# Shranjevanje obdelanih podatkov
hourly_data.to_csv('data/processed/processed_data.csv')