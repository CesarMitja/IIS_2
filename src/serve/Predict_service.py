from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.linear(hn[-1])
        return out

app = Flask(__name__)
CORS(app)
# Nastavitev poti do modela in scalerja
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(base_dir, 'models', 'rnn_model.pth')
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
csv_file_path = os.path.join(base_dir, 'data','processed', 'data_for_prediction.csv')

# Nalaganje modela in scalerja
model = RNNModel(input_size=9, hidden_size=50, num_layers=1, output_size=10)  # Preverite te številke
model.load_state_dict(torch.load(model_path))
model.eval()

scaler = joblib.load(scaler_path)

# Definicija funkcije za napovedovanje
def make_prediction(input_data):
    # Obdelava in priprava vhodnih podatkov
    data_scaled = scaler.transform(input_data)
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)

    # Napoved
    with torch.no_grad():
        prediction = model(data_tensor)
    prediction = prediction.numpy().flatten()

    # Deskaliranje samo za 'available_bike_stands'
    dummy_output = np.zeros((10, input_data.shape[1]))
    available_bike_stands_index = list(input_data.columns).index('available_bike_stands')
    dummy_output[:, available_bike_stands_index] = prediction

    prediction_rescaled = scaler.inverse_transform(dummy_output)[:, available_bike_stands_index]

    return prediction_rescaled

def load_last_72_rows(csv_file_path):
    # Preberemo celotno datoteko
    df = pd.read_csv(csv_file_path)
    
    # Preverimo, če ima datoteka vsaj 72 vrstic
    if len(df) >= 72:
        # Vzamemo zadnjih 72 vrstic
        df = df.tail(72)
    else:
        raise ValueError("Datoteka nima dovolj vrstic (potrebujemo vsaj 72 vrstic).")
    
    return df

# Flask ruta za napovedovanje
@app.route('/predict', methods=['POST'])
def predict():
    
    input_data = load_last_72_rows(csv_file_path)
    # Napoved in odstranitev datuma, če ni potreben za model
    dates = input_data.pop('date')
    prediction = make_prediction(input_data)

    # Priprava in vračanje rezultatov
    results = pd.DataFrame({
        'hour_ahead': np.arange(1, 11), 
        'prediction': prediction
    })
    return jsonify(results.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)