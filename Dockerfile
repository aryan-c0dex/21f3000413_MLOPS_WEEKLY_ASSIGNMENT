FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY mlruns/ ./mlruns/

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

EXPOSE 5001

CMD ["python", "src/inference.py"]
