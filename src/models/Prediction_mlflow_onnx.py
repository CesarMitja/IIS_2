import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

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
train_data_path = os.path.join(base_dir, 'data/processed', 'train.csv')
test_data_path = os.path.join(base_dir, 'data/processed', 'test.csv')
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
model_path = os.path.join(base_dir, 'models', 'rnn_model.pth')
onnx_model_path = os.path.join(base_dir, 'models', 'model.onnx')

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
 



train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])
for df in [train_data, test_data]:
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

features = ['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands', 'hour', 'day_of_week', 'month']
#features2 = ['temperature','bike_stands', 'available_bike_stands', 'hour', 'day_of_week', 'month']

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

train_features = pipeline.fit_transform(train_data[features])
test_features = pipeline.transform(test_data[features])

class WeightedMSELoss(nn.Module):
    def __init__(self, start_hour=22, end_hour=3, weight_multiplier=3):
        super(WeightedMSELoss, self).__init__()
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.weight_multiplier = weight_multiplier

    def forward(self, outputs, targets, hours):
        mse_loss = (outputs - targets) ** 2

        weights = torch.ones_like(targets)
        critical_hours_mask = ((hours >= self.start_hour) & (hours <= 22)) | ((hours >= 1) & (hours <= self.end_hour))
        weights[critical_hours_mask] = self.weight_multiplier

        weighted_loss = mse_loss * weights
        return weighted_loss.mean()

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, (hn, _) = self.lstm(x)
        return self.linear(hn[-1])

model = RNNModel(input_size=len(features), hidden_size=40, num_layers=2, output_size=7, dropout=0.1)
criterion = WeightedMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
sequence_length = 50
forecast_horizon = 7

def create_sequences_with_hours(input_data, target_data, hours_data, sequence_length, forecast_horizon):
    sequences, targets, hours = [], [], []
    for start_pos in range(len(input_data) - sequence_length - forecast_horizon + 1):
        end_pos = start_pos + sequence_length
        seq_hours = hours_data[start_pos:end_pos]  # Grab the sequence of hours
        sequences.append(input_data[start_pos:end_pos])
        targets.append(target_data[end_pos:end_pos + forecast_horizon])
        hours.append(seq_hours[-1])  # We only need the last hour for the corresponding target
    return np.array(sequences), np.array(targets), np.array(hours)

train_hours = train_data['hour'].values
test_hours = test_data['hour'].values

X_train, y_train, train_hours_seq = create_sequences_with_hours(train_features, train_data['available_bike_stands'].values, train_hours, sequence_length, forecast_horizon)
X_test, y_test, test_hours_seq = create_sequences_with_hours(test_features, test_data['available_bike_stands'].values, test_hours, sequence_length, forecast_horizon)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
train_hours_seq = torch.tensor(train_hours_seq, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
test_hours_seq = torch.tensor(test_hours_seq, dtype=torch.float32)

train_data = TensorDataset(X_train, y_train, train_hours_seq)
test_data = TensorDataset(X_test, y_test, test_hours_seq)

train_loader = DataLoader(train_data, batch_size=200, shuffle=True)
test_loader = DataLoader(test_data, batch_size=200, shuffle=False)

def train_and_evaluate_model():
    with mlflow.start_run():
        global best_test_loss
        for epoch in range(50):
            model.train()
            total_train_loss = 0
            for inputs, targets, hours in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.view(targets.size(0), -1), hours)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
                optimizer.step()
                total_train_loss += loss.item() * inputs.size(0)
            avg_train_loss = total_train_loss / len(train_loader.dataset)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            print("Epoch:", epoch, "train_loss = ", avg_train_loss)

            scheduler.step() 


            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for inputs, targets, hours in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.view(targets.size(0), -1), hours)
                    total_test_loss += loss.item() * inputs.size(0)
            avg_test_loss = total_test_loss / len(test_loader.dataset)
            mlflow.log_metric("test_loss", avg_test_loss)
            print(f'Test Loss: {avg_test_loss}')

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), model_path)
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
