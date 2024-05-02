from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import requests
import pymongo
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS
import onnxruntime as ort
from mlflow.tracking import MlflowClient
import mlflow
import dagshub
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from flask_cors import cross_origin

app = Flask(__name__)


CONNECTION_STRING = "mongodb+srv://cesi:Hondacbr125.@ptscluster.gkdlocr.mongodb.net/?retryWrites=true&appName=PTScluster"
client = pymongo.MongoClient(CONNECTION_STRING)
# Connect to the database
db = client.IIS
# Assuming you have a collection named 'predictions'
predictions_collection = db.iis2
actual_collection = db.iis2_a

dagshub_token = '9afb330391a28d5362f1f842cac05eef42708362'
dagshub.auth.add_app_token(dagshub_token)
dagshub.init(repo_name="IIS_2", repo_owner="CesarMitja", mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/CesarMitja/IIS_2.mlflow')
mlflow.set_experiment('Bike_Stand_Prediction_Service_ONNX_Daily')




CORS(app, resources={r"/*": {"origins": "*"}}) 

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
@cross_origin(origin='*')  # Adjust as needed
def predict():
    try:
        input_data = load_last_72_rows(csv_file_path)
        predictions = make_prediction(input_data)
        results = {"predictions": predictions.tolist()}
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})
    


def predict1():
    try:
        input_data = load_last_72_rows(csv_file_path)
        predictions = make_prediction(input_data)
        results = {"predictions": predictions.tolist()}
        doc = {
            "timestamp": datetime.datetime.utcnow(),
            "predictions": predictions.tolist()}
        predictions_collection.insert_one(doc)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})
    
def calculate_metrics():
    yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    start_of_day = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = yesterday.replace(hour=23, minute=59, second=59, microsecond=999)
    
    records = db.iis2.find = {
        "timestamp": {"$gte": start_of_day, "$lt": end_of_day}
    }
    records2 = db.iis2_a.find = {
        "timestamp": {"$gte": start_of_day, "$lt": end_of_day}
    }    
    predictions = []
    actuals = []
    
    for record in records:
        predictions.append(record['predictions'][0])
    for record1 in records2:
        actuals.append(record1['actual'])
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    if len(actuals) > 0 and len(predictions) > 0:
        mse = mean_squared_error(actuals, predictions)
        print(f"Calculated MSE for {yesterday.strftime('%Y-%m-%d')}: {mse}")
        
        with mlflow.start_run():
            mlflow.log_metric("daily_mse", mse)
    else:
        print("No sufficient data for metric calculation.")

def save_data():
    API_URL = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
    TARGET_STATION_NAME = "GOSPOSVETSKA C. - TURNERJEVA UL."
    response = requests.get(API_URL)
    response.raise_for_status()
    stations = response.json()
    for station in stations:
        if station['name'] == TARGET_STATION_NAME:
            df1 = pd.DataFrame([station])
    df2 = df1['available_bike_stands'].values
    df2 = df2[0]
    doc = {
        "timestamp": datetime.datetime.utcnow(),
        "actual": df2.tolist()}
    actual_collection.insert_one(doc)






scheduler = BackgroundScheduler()
scheduler.add_job(func=calculate_metrics, trigger='cron', hour=0)
scheduler.add_job(func=predict1, trigger='cron', minute=0)
scheduler.add_job(func=save_data, trigger='cron', minute=0)
scheduler.start()

if __name__ == '__main__':
    app.run(debug=True)
