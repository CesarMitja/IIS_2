FROM python:3.9

WORKDIR /app

RUN pip install --no-cache-dir flask numpy scikit-learn joblib tensorflow pandas flask_cors

COPY . .

ENV FLASK_APP=App.py
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 5000

CMD ["flask", "run"]
