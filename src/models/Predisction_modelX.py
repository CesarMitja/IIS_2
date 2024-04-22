import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

#Poti do datotek 
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(base_dir, 'data', 'processed', 'data_for_prediction.csv')
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
model_path = os.path.join(base_dir, 'models', 'rnn_model.pth')
train_metrics_path = os.path.join(base_dir, 'reports', 'train_metrics.txt')
test_metrics_path = os.path.join(base_dir, 'reports', 'test_metrics.txt')

data = pd.read_csv(data_path, parse_dates=['date'])
features = ['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature',
            'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands']

scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

joblib.dump(scaler, scaler_path)

def create_sequences(data, input_width, forecast_horizon):
    X = []
    y = []
    for i in range(len(data) - input_width - forecast_horizon + 1):
        X.append(data[i:(i+input_width), :])
        y.append(data[(i+input_width):(i+input_width+forecast_horizon), -1])  
    return np.array(X), np.array(y)

input_width = 72
forecast_horizon = 10

X, y = create_sequences(data[features].values, input_width, forecast_horizon)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.linear(hn[-1])
        return out

model = RNNModel(input_size=len(features), hidden_size=50, num_layers=1, output_size=forecast_horizon)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(targets.size(0), -1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss}')
    return train_losses

def evaluate_model(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(targets.size(0), -1))
            test_loss += loss.item() * inputs.size(0)
        test_loss /= len(test_loader.dataset)
    return test_loss

train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=20)

test_loss = evaluate_model(model, test_loader, criterion)
print(f'Test Loss: {test_loss}')

torch.save(model.state_dict(), model_path)

with open(train_metrics_path, 'w') as f:
    f.write(f'Training Losses: {train_losses}\n')
with open(test_metrics_path, 'w') as f:
    f.write(f'Test Loss: {test_loss}\n')