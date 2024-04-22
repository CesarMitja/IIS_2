FROM python:3.9

WORKDIR /app

RUN pip install -r requirements.txt

COPY . .

ENV FLASK_APP=src/serve/Predict_service.py
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 5000

CMD ["flask", "run"]
