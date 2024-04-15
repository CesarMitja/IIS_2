from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import os


current_directory = os.path.dirname(__file__)

# Nastavitev poti do modela in scaler datoteke
model_path = os.path.join(current_directory, 'best_model_GRU.h5')
scaler_path = os.path.join(current_directory, 'scaler6.pkl')
# Naložitev modela in scaler-ja
#model = tf.keras.models.load_model('C:/Users/mitja/Desktop/inžinirstvo/Prva naloga/models/best_model_GRU.h5')
#scaler = joblib.load('C:/Users/mitja/Desktop/inžinirstvo/Prva naloga/models/scaler6.pkl')
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)
app = Flask(__name__)

@app.route('/napoved/mbajk', methods=['POST'])
def napoved_naloga6():
    data = request.get_json()
    timeseries = data.get('timeseries')

    if len(timeseries) < 186:
        return jsonify({'error': 'Invalid timeseries length (too short). Expected 186 data points'}), 400
    if len(timeseries) > 186:
        return jsonify({'error': 'Invalid timeseries length (too long). Expected 186 data points'}), 400

    timeseries_reshaped = np.array([timeseries]).reshape(1, 1, 186)
    prediction = model.predict(timeseries_reshaped)
    prediction_descaled = scaler.inverse_transform(prediction.reshape(-1, 1))
    return jsonify({'prediction': prediction_descaled.tolist()})

if __name__ == '__main__':
    app.run(debug=True)