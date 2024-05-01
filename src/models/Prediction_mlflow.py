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

dagshub_token = os.environ.get('DAGS')
dagshub.auth.add_app_token(dagshub_token)
dagshub.init(repo_name="IIS-2", repo_owner="CesarMitja", mlflow=True)
# MLflow settings
mlflow.set_tracking_uri('https://dagshub.com/CesarMitja/IIS_2.mlflow')
mlflow.set_experiment('Bike_Stand_Prediction')

# File paths
base_dir = os.path.abspath('.')
train_data_path = os.path.join(base_dir, 'data/processed', 'train.csv')
test_data_path = os.path.join(base_dir, 'data/processed', 'test.csv')
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
model_path = os.path.join(base_dir, 'models', 'rnn_model.pth')

# Data reading and preparation
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
features = ['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 
            'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands']

# Preprocessing pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# Data preprocessing
train_features = pipeline.fit_transform(train_data[features])
test_features = pipeline.transform(test_data[features])
joblib.dump(pipeline, scaler_path)

# Define the model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.linear(hn[-1])
        return out.squeeze()

model = RNNModel(input_size=len(features), hidden_size=100, num_layers=1, output_size=1)

# Training settings
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation function with MLflow
def train_and_evaluate_model(model, train_features, test_features, criterion, optimizer, num_epochs=20):
    with mlflow.start_run() as run:
        # Train the model
        for epoch in range(num_epochs):
            model.train()
            outputs = model(torch.tensor(train_features, dtype=torch.float32))
            train_targets = torch.tensor(train_data['available_bike_stands'].values, dtype=torch.float32).view(-1, 1)
            loss = criterion(outputs, train_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mlflow.log_metric("train_loss", loss.item(), step=epoch)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(test_features, dtype=torch.float32))
            test_targets = torch.tensor(test_data['available_bike_stands'].values, dtype=torch.float32).view(-1, 1)
            test_loss = criterion(predictions, test_targets)
            mlflow.log_metric("test_loss", test_loss.item())

        # Log and register the model if it's an improvement or if no model exists
        mlflow.pytorch.log_model(model, "model", registered_model_name="Bike_Stand_Prediction_Model")

train_and_evaluate_model(model, train_features, test_features, criterion, optimizer)
