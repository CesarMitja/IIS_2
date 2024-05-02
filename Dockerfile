FROM python:3.9

WORKDIR /app

RUN pip install --no-cache-dir poetry pytest flask pandas torch flask_cors joblib joblib scikit-learn numpy mlflow apscheduler onnxruntime dagshub datetime PyMongo pymongo

COPY . .

ENV FLASK_APP=src/serve/Predict_6.py
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 5000

CMD ["flask", "run"]
