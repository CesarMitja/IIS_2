from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib

# Naložitev modela in scaler-ja
model = tf.keras.models.load_model('C:/Users/mitja/Desktop/inžinirstvo/Prva naloga/models/best_model_GRU.h5')
scaler = joblib.load('C:/Users/mitja/Desktop/inžinirstvo/Prva naloga/models/scaler6.pkl')

app = Flask(__name__)

@app.route('/mbajk/predict/', methods=['POST'])
def predict():
    data = request.json
    # Predvidevamo, da bo vhodni podatek vseboval 'last_update' in 'available_bike_stands'
    df = pd.DataFrame([data])
    df['last_update'] = pd.to_datetime(df['last_update'])
    df.set_index('last_update', inplace=True)

    # Predpriprava podatkov
    input_data = scaler.transform(df[['available_bike_stands']])
    input_data = input_data.reshape(-1, 1, 1)  # Prilagoditev oblike za LSTM

    # Napoved
    prediction = model.predict(input_data)
    prediction = scaler.inverse_transform(prediction)  # Vrnitev napovedi v prvotno lestvico

    return jsonify({"prediction": int(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)