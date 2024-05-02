import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
# Setup MLflow
mlflow.set_tracking_uri('https://dagshub.com/CesarMitja/IIS_2.mlflow')
mlflow.set_experiment('Bike_Stand_Prediction')

client = MlflowClient()
try:
    # Attempt to load the latest registered model's metrics
    latest_version = client.get_latest_versions("Bike_Stand_Prediction_Model", stages=["Production"])[0]
    best_test_loss = client.get_metric_history(latest_version.run_id, "test_loss")[-1].value
except (IndexError, ValueError, mlflow.exceptions.MlflowException):
    best_test_loss = float('inf')  # If no model exists, set to infinity

# File paths
base_dir = os.path.abspath('.')
train_data_path = os.path.join(base_dir, 'data/processed', 'train.csv')
test_data_path = os.path.join(base_dir, 'data/processed', 'test.csv')
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
model_path = os.path.join(base_dir, 'models', 'rnn_model.pth')

# Read and prepare data
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
features = ['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 
            'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands']

# Preprocessing pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# Preprocess data
train_features = pipeline.fit_transform(train_data[features])
test_features = pipeline.transform(test_data[features])


# Define the model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.linear(hn[-1])

model = RNNModel(input_size=len(features), hidden_size=50, num_layers=1, output_size=7)

# Training settings
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to create data sequences
def create_sequences(input_data, target_data, sequence_length, forecast_horizon):
    sequences = []
    targets = []
    for start_pos in range(len(input_data) - sequence_length - forecast_horizon + 1):
        end_pos = start_pos + sequence_length
        seq = input_data[start_pos:end_pos]
        target_seq = target_data[end_pos:end_pos + forecast_horizon]
        sequences.append(seq)
        targets.append(target_seq)
    return np.array(sequences), np.array(targets)

# Update sequence creation calls in your main script
sequence_length = 32  # 72 hours of data
forecast_horizon = 7  # Predict 7 hours ahead

X_train, y_train = create_sequences(train_features, train_data['available_bike_stands'].values, sequence_length, forecast_horizon)
X_test, y_test = create_sequences(test_features, test_data['available_bike_stands'].values, sequence_length, forecast_horizon)
print(f"Train dataset size: {len(X_train)}, Test dataset size: {len(X_test)}")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Train and evaluate model with MLflow
def train_and_evaluate_model():
    with mlflow.start_run():
        for epoch in range(20):  # Train for 20 epochs
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
            mlflow.pytorch.log_model(model, "model", registered_model_name="Bike_Stand_Prediction_Model")
            joblib.dump(pipeline, scaler_path)
            print("New model saved with improved test loss: {:.4f}".format(avg_test_loss))
        else:
            print("No improvement in test loss: {:.4f}".format(avg_test_loss))

train_and_evaluate_model()