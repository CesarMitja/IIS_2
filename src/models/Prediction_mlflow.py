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

# Setup MLflow
mlflow.set_tracking_uri('https://dagshub.com/CesarMitja/IIS_2.mlflow')
mlflow.set_experiment('Bike_Stand_Prediction')

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
joblib.dump(pipeline, scaler_path)

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
def create_sequences(input_data, target_data, sequence_length):
    sequences = []
    targets = []
    for start_pos in range(len(input_data) - sequence_length):
        end_pos = start_pos + sequence_length
        sequence = input_data[start_pos:end_pos]
        target = target_data[end_pos:end_pos+7]  # Adjust target to predict 7 steps ahead
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Create sequences
sequence_length = 72  # 72 hours input
X_train, y_train = create_sequences(train_features, train_data['available_bike_stands'].values, sequence_length)
X_test, y_test = create_sequences(test_features, test_data['available_bike_stands'].values, sequence_length)

X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Train and evaluate model with MLflow
def train_and_evaluate_model():
    with mlflow.start_run():
        for epoch in range(20):  # 20 epochs
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                mlflow.log_metric("train_loss", loss.item(), step=epoch)

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
            avg_loss = total_loss / len(test_loader)
            mlflow.log_metric("test_loss", avg_loss)
            print(f'Test Loss: {avg_loss}')

        # Save and log the model
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "model", registered_model_name="Bike_Stand_Prediction_Model")

train_and_evaluate_model()