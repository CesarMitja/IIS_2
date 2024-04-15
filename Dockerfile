# Uporabi uradno Python sliko kot osnovno sliko
FROM python:3.9-slim

# Nastavi delovni direktorij v kontejnerju
WORKDIR /app

# Kopiraj 'pyproject.toml' (in potencialno 'poetry.lock', če obstaja) v delovni direktorij
COPY pyproject.toml poetry.lock* /app/

# Namesti 'poetry' in odvisnosti projekta
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

# Kopiraj vse aplikacijske datoteke v kontejner
COPY . /app

# Nastavi spremenljivke okolja za Flask
ENV FLASK_APP notebooks/App.py
ENV FLASK_RUN_HOST 0.0.0.0

# Izpostavi port, ki ga Flask uporablja (privzeto 5000, razen če spremenite)
EXPOSE 5000

# Ukaz za zagon Flask aplikacije
CMD ["flask", "run"]