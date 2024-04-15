# Uporabi uradno Python sliko kot osnovno sliko
FROM python:3.9

# Nastavi delovni direktorij v kontejnerju
WORKDIR /app


# Namesti odvisnosti s pip namesto poetry, ker TensorFlow povzroča težave
RUN pip install flask numpy scikit-learn joblib tensorflow pandas flask_cors

# Kopiraj preostale aplikacijske datoteke v kontejner
COPY . .

# Nastavi spremenljivke okolja za Flask
ENV FLASK_APP=notebooks/App.py
ENV FLASK_RUN_HOST=0.0.0.0

# Izpostavi port, ki ga Flask uporablja (privzeto 5000, razen če spremenite)
EXPOSE 5000

# Ukaz za zagon Flask aplikacije
CMD ["flask", "run"]
