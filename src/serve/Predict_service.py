from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, (hn, _) = self.lstm(x)
        x = self.linear(hn[-1])
        return x

app = Flask(__name__)
CORS(app)

# Update paths accordingly
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(base_dir, 'models', 'rnn_model.pth')
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
csv_file_path = os.path.join(base_dir, 'data', 'processed', 'data_for_prediction.csv')

features = ['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 
            'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands']

model = RNNModel(input_size=len(features), hidden_size=50, num_layers=1, output_size=7)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

scaler = joblib.load(scaler_path)

def make_prediction(input_data):
    input_data = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        prediction = model(input_tensor)
    prediction = prediction.numpy().flatten()

    return prediction

def load_last_72_rows(csv_file_path):
    df = pd.read_csv(csv_file_path)
    features = ['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 
            'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands']
    df = df[features]  # Ensure the correct columns are used
    if len(df) >= 32:
        df = df.tail(32)  # Use the last 72 hours
    else:
        raise ValueError("Not enough data. Need at least 72 rows.")
    
    return df#.values  # Return as NumPy array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = load_last_72_rows(csv_file_path)
        #print(input_data)
        predictions = make_prediction(input_data)
        results = {"predictions": predictions.tolist()}
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
