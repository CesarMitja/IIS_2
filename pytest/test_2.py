import pytest
import requests
import os
from flask import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.serve.Predict_service import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_success(client):
    response = client.post('/predict', content_type='application/json')
    data = json.loads(response.text)
    assert len(data) > 0

def test_predict_failure(client):
    response = client.post('/predict', content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)

def test_mbajk_api():
    API_URL = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
    response = requests.get(API_URL)
    assert response.status_code == 200

def test_mateo_api():
    WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
    WEATHER_PARAMS = {
    'latitude': 46.5547,
    'longitude': 15.6467,
    'hourly': 'temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,surface_pressure'
    }
    response = requests.get(WEATHER_API_URL, params=WEATHER_PARAMS)
    assert response.status_code == 200
    