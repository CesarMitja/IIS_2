import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.onnx
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

dagshub_token = '9afb330391a28d5362f1f842cac05eef42708362'
dagshub.auth.add_app_token(dagshub_token)
dagshub.init(repo_name="IIS_2", repo_owner="CesarMitja", mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/CesarMitja/IIS_2.mlflow')
mlflow.set_experiment('Bike_Stand_Prediction_ONNX')

client = MlflowClient()
try:
    latest_version = client.get_latest_versions("Bike_Stand_Prediction_Model_ONNX")[0]
    best_test_loss = client.get_metric_history(latest_version.run_id, "test_loss")[-1].value
except (IndexError, ValueError, mlflow.exceptions.MlflowException):
    best_test_loss = float('inf')

base_dir = os.path.abspath('.')
train_data_path = os.path.join(base_dir, 'data/processed', 'data_for_prediction.csv')
test_data_path = os.path.join(base_dir, 'data/processed', 'test.csv')
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
model_path = os.path.join(base_dir, 'models', 'rnn_model.pth')
onnx_model_path = os.path.join(base_dir, 'models', 'model.onnx')

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
features = ['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands']

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])
train_features = pipeline.fit_transform(train_data[features])
test_features = pipeline.transform(test_data[features])

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x, (hn, _) = self.lstm(x)
        return self.linear(hn[-1])

model = RNNModel(input_size=len(features), hidden_size=100, num_layers=2, output_size=7, dropout=0.3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def create_sequences(input_data, target_data, sequence_length, forecast_horizon):
    sequences, targets = [], []
    for start_pos in range(len(input_data) - sequence_length - forecast_horizon + 1):
        end_pos = start_pos + sequence_length
        sequences.append(input_data[start_pos:end_pos])
        targets.append(target_data[end_pos:end_pos + forecast_horizon])
    return np.array(sequences), np.array(targets)

sequence_length = 72
forecast_horizon = 7
X_train, y_train = create_sequences(train_features, train_data['available_bike_stands'].values, sequence_length, forecast_horizon)
X_test, y_test = create_sequences(test_features, test_data['available_bike_stands'].values, sequence_length, forecast_horizon)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

def train_and_evaluate_model():
    with mlflow.start_run():
        for epoch in range(50):
            model.train()
            total_train_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.view(targets.size(0), -1))
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * inputs.size(0)
            avg_train_loss = total_train_loss / len(train_loader.dataset)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        model.eval()
        total_test_loss = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(targets.size(0), -1))
            total_test_loss += loss.item() * inputs.size(0)
        avg_test_loss = total_test_loss / len(test_loader.dataset)
        mlflow.log_metric("test_loss", avg_test_loss)
        print(f'Test Loss: {avg_test_loss}')

        if avg_test_loss < best_test_loss:
            torch.save(model.state_dict(), model_path)
            # Export to ONNX and save the model to MLflow
            dummy_input = torch.randn(1, sequence_length, len(features))
            torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True, opset_version=11)
            mlflow.log_artifact(onnx_model_path, "model")
            mlflow.pytorch.log_model(model, "model", registered_model_name="Bike_Stand_Prediction_Model_ONNX")
            joblib.dump(pipeline, scaler_path)
            mlflow.log_artifact(scaler_path, "model")
            print("New model saved with improved test loss: {:.4f}".format(avg_test_loss))
        else:
            print("No improvement in test loss: {:.4f}".format(avg_test_loss))

train_and_evaluate_model()
