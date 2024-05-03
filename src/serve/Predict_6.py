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
import subprocess
app = Flask(__name__)


CONNECTION_STRING = "mongodb+srv://cesi:Hondacbr125.@ptscluster.gkdlocr.mongodb.net/?retryWrites=true&appName=PTScluster"
client = pymongo.MongoClient(CONNECTION_STRING)

db = client.IIS

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


def update_data():
    subprocess.run(['dvc', 'pull', '-r','origin', 'data/data_for_prediction.csv'], capture_output=True, text=True)



def load_model(run_id, artifact_path):
    local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact_path}")
    ort_session = ort.InferenceSession(local_model_path)
    return ort_session


def load_scaler(run_id, artifact_path):
    local_scaler_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact_path}")
    return joblib.load(local_scaler_path)

model = load_model(latest_version.run_id, "model/model.onnx")
scaler = load_scaler(latest_version.run_id, "model/scaler.pkl")

features = ['date','temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 
            'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands']


features1 = ['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands', 'hour', 'day_of_week', 'month']

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_file_path = os.path.join(base_dir, 'data', 'processed', 'data_for_prediction.csv')

def make_prediction(input_data):
    print("make 1")
    input_data['date'] = pd.to_datetime(input_data['date'])
    print("make 2")
    for df in [input_data ]:
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
    print("make 3")
    input_data.pop('date')
    input_data = scaler.transform(input_data)

    input_tensor = np.array(input_data, dtype=np.float32).reshape(1, -1, len(features1))  

    ort_inputs = {model.get_inputs()[0].name: input_tensor}
    ort_outs = model.run(None, ort_inputs)

    return ort_outs[0].flatten()

def load_last_72_rows(csv_file_path):
    print("72 prvi")
    df = pd.read_csv(csv_file_path)
    df = df[features]
    if len(df) >= 72:
        df = df.tail(72)
        print("72 notri")
    else:
        raise ValueError("Not enough data. Need at least 72 rows.")
    return df


@app.route('/predict', methods=['POST'])
@cross_origin(origin='*')  
def predict():
    try:
        update_data()
        print("prvi")
        input_data = load_last_72_rows(csv_file_path)
        print("drugi")
        predictions = make_prediction(input_data)
        print("tretji")
        results = {"predictions": predictions.tolist()}
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})
    


def predict1():
    try:
        update_data()
        input_data = load_last_72_rows(csv_file_path)
        predictions = make_prediction(input_data)
        doc = {
            "timestamp": datetime.datetime.utcnow(),
            "predictions": predictions.tolist()}
        predictions_collection.insert_one(doc)
        return True
    except Exception as e:
        return False
    
def calculate_metrics():
    yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    start_of_day = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = yesterday.replace(hour=23, minute=59, second=59, microsecond=999)

    predictions_records = list(db.iis2.find({
        "timestamp": {"$gte": start_of_day, "$lt": end_of_day}
    }))
    actuals_records = list(db.iis2_a.find({
        "timestamp": {"$gte": start_of_day, "$lt": end_of_day}
    }))

    # Preparing data containers
    actuals = {record['timestamp'].replace(minute=0, second=0, microsecond=0): record['actual']
            for record in actuals_records}

    # Calculate MSE for each set of predictions
    for record in predictions_records:
        base_time = record['timestamp']
        if isinstance(base_time, int):  # Convert from UNIX timestamp if necessary
            base_time = datetime.datetime.utcfromtimestamp(base_time / 1000)
        base_time = base_time.replace(minute=0, second=0, microsecond=0)
        
        prediction_values = [pred for pred in record['predictions']]
        mse_values = []
        
        with mlflow.start_run():
            for i, prediction in enumerate(prediction_values):
                pred_time = base_time + datetime.timedelta(hours=i)
                if pred_time in actuals:
                    mse = mean_squared_error([prediction], [actuals[pred_time]])
                    mse_values.append(mse)
                    mlflow.log_metric(f'prediction_{base_time.hour:02d}_{i}_mse', mse)
            
            if mse_values:
                avg_mse = np.mean(mse_values)
                mlflow.log_metric(f'prediction_{base_time.hour:02d}_avg_mse', avg_mse)
                print(f"Average MSE for predictions starting at {base_time.hour:02d}:00: {avg_mse}")
            else:
                print(f"No matching actual data for predictions starting at {base_time.hour:02d}:00.")

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
scheduler.add_job(func=save_data)
scheduler.start()

if __name__ == '__main__':
    app.run(debug=True)
