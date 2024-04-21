import pytest
from flask import json
from app import app  # Import your Flask app from the file where it is defined

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_success(client):
    # To je simulacija testnih podatkov, ki ustrezajo strukturi vaše datoteke
    sample_data = {
        "data": [
            {"date": "2024-03-26T17:00:00", "temperature": 12.3, "relative_humidity": 38, "dew_point": -1.6, "apparent_temperature": 8.1, "precipitation_probability": 0, "rain": 0.0, "surface_pressure": 970.5, "bike_stands": 22, "available_bike_stands": 10},
            # Dodajte še 71 vrstic podatkov za popoln test
        ]
    }
    response = client.post('/predict', data=json.dumps(sample_data), content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert len(data['prediction']) == 10  # Preverite, če napoved vrne 10 ur naprej

def test_predict_failure(client):
    # Test z nepravilnimi podatki, ki ne vsebujejo zahtevanih polj
    wrong_data = {
        "data": [
            {"date": "2024-03-26T17:00:00", "temperature": 12.3}
        ]
    }
    response = client.post('/predict', data=json.dumps(wrong_data), content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data