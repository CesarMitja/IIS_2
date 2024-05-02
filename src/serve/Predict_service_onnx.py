from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS
import onnxruntime as ort
from mlflow.tracking import MlflowClient
import mlflow
import dagshub

dagshub_token = '9afb330391a28d5362f1f842cac05eef42708362'
dagshub.auth.add_app_token(dagshub_token)
dagshub.init(repo_name="IIS_2", repo_owner="CesarMitja", mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/CesarMitja/IIS_2.mlflow')
mlflow.set_experiment('Bike_Stand_Prediction_Service')

app = Flask(__name__)
CORS(app)

mlflow_client = MlflowClient()
model_name = "Bike_Stand_Prediction_Model_ONNX"
latest_version = mlflow_client.get_latest_versions(model_name)[0]

# Function to download and load the ONNX model
def load_model(run_id, artifact_path):
    local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact_path}")
    ort_session = ort.InferenceSession(local_model_path)
    return ort_session

# Function to download and load the scaler
def load_scaler(run_id, artifact_path):
    local_scaler_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact_path}")
    return joblib.load(local_scaler_path)

model = load_model(latest_version.run_id, "model/model.onnx")
scaler = load_scaler(latest_version.run_id, "model/scaler.pkl")

features = ['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 
            'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands']

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_file_path = os.path.join(base_dir, 'data', 'processed', 'data_for_prediction.csv')

def make_prediction(input_data):
    # First transform the input data using the scaler
    input_data = scaler.transform(input_data)

    input_tensor = np.array(input_data, dtype=np.float32).reshape(1, -1, len(features))  

    ort_inputs = {model.get_inputs()[0].name: input_tensor}
    ort_outs = model.run(None, ort_inputs)

    return ort_outs[0].flatten()

def load_last_72_rows(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df = df[features]
    if len(df) >= 72:
        df = df.tail(72)
    else:
        raise ValueError("Not enough data. Need at least 72 rows.")
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = load_last_72_rows(csv_file_path)
        predictions = make_prediction(input_data)
        results = {"predictions": predictions.tolist()}
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
