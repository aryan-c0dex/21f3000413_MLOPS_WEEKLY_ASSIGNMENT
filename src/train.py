import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import os
import subprocess

# Bucket name
BUCKET_NAME = "21f3000413-mlops-week1-bucket"

# Load dataset from GCS
data_path = f"gs://{BUCKET_NAME}/data/iris.csv"
data = pd.read_csv(data_path)

# Split data
X = data.drop("species", axis=1)
y = data["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"Model Accuracy: {accuracy}")

# Create timestamp folder
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
local_model_path = "model.pkl"
local_metrics_path = "metrics.txt"

# Save locally
joblib.dump(model, local_model_path)

with open(local_metrics_path, "w") as f:
    f.write(f"Accuracy: {accuracy}")

# Upload to GCS
artifact_path = f"gs://{BUCKET_NAME}/artifacts/{timestamp}/"

subprocess.run(["gsutil", "cp", local_model_path, artifact_path])
subprocess.run(["gsutil", "cp", local_metrics_path, artifact_path])

print(f"Artifacts stored at: {artifact_path}")
