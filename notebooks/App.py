from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib

# Naložitev modela in scaler-ja
model = tf.keras.models.load_model('C:/Users/mitja/Desktop/inžinirstvo/Prva naloga/models/best_model_GRU.h5')
scaler = joblib.load('C:/Users/mitja/Desktop/inžinirstvo/Prva naloga/models/scaler6.pkl')

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