import pytest
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