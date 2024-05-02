import shutil
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import torch.onnx
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS
import torch.nn as nn
from mlflow.tracking import MlflowClient
import mlflow
import dagshub


dagshub_token = '9afb330391a28d5362f1f842cac05eef42708362'
dagshub.auth.add_app_token(dagshub_token)
dagshub.init(repo_name="IIS_2", repo_owner="CesarMitja", mlflow=True)
# Setup MLflow
mlflow.set_tracking_uri('https://dagshub.com/CesarMitja/IIS_2.mlflow')
mlflow.set_experiment('Bike_Stand_Prediction_Service')

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

mlflow_client = MlflowClient()
features = ['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 
            'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands']

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_file_path = os.path.join(base_dir, 'data', 'processed', 'data_for_prediction.csv')
#scaler = joblib.load(os.path.join(base_dir,'models','scaler.pkl'))




def download_and_load_model1(run_id, artifact_path):
    # Ensure the model is downloaded to a specific path you have full control over
    local_model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    safe_model_path = "models/novi.pth"  # Specify your path
    shutil.copy(local_model_path, safe_model_path)
    return torch.load(safe_model_path)

def download_and_load_model2(run_id, artifact_path):
    # Ensure the model is downloaded to a specific path you have full control over
    local_model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    safe_model_path = "models/novi1.pkl"  # Specify your path
    shutil.copy(local_model_path, safe_model_path)
    return joblib.load(safe_model_path)

model_name = "Bike_Stand_Prediction_Model"
latest_version = mlflow_client.get_latest_versions(model_name)[0]
model_path = "model/data/model.pth"  # Adjust based on your artifact structure
model = download_and_load_model1(latest_version.run_id, model_path)
scaler_path = "model/scaler.pkl"  # Adjust based on your artifact structure
#scaler = joblib.load(scaler_path)
scaler = download_and_load_model2(latest_version.run_id, scaler_path)
model.eval()



# def load_artifact(run_id, artifact_file):
#     artifact_path = mlflow_client.download_artifacts(run_id, artifact_file)
#     return artifact_path

# # Fetch the latest model and scaler
# model_name = "Bike_Stand_Prediction_Model"
# latest_version = mlflow_client.get_latest_versions(model_name)[0]

# model_path = load_artifact(latest_version.run_id, "data/model.pth")
# scaler_path = load_artifact(latest_version.run_id, "model/scaler.pkl")

# model = mlflow.pytorch.load_model(f"runs:/{latest_version.run_id}/model")
# scaler = joblib.load(scaler_path)

# model.eval()



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
