import pandas as pd
import joblib
import sys
import subprocess
from sklearn.metrics import accuracy_score

BUCKET_NAME = "21f3000413-mlops-week1-bucket"

if len(sys.argv) != 2:
    print("Usage: python src/inference.py <timestamp>")
    sys.exit(1)

timestamp = sys.argv[1]

# Download model from GCS
gcs_model_path = f"gs://{BUCKET_NAME}/artifacts/{timestamp}/model.pkl"
local_model_path = "model.pkl"

subprocess.run(["gsutil", "cp", gcs_model_path, local_model_path])

# Load model
model = joblib.load(local_model_path)

# Load evaluation data
data_path = f"gs://{BUCKET_NAME}/data/iris.csv"
data = pd.read_csv(data_path)

X = data.drop("species", axis=1)
y = data["species"]

preds = model.predict(X)
accuracy = accuracy_score(y, preds)

print(f"Inference Accuracy using model from {timestamp}: {accuracy}")
