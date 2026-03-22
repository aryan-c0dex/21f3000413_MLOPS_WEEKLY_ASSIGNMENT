from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import os

app = Flask(__name__)

# Load model directly from mlruns folder
model = mlflow.sklearn.load_model("mlruns/1/models/m-9c18106a16294da48859a049f6544020/artifacts")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sample = pd.DataFrame([{
        "sepal_length": data["sepal_length"],
        "sepal_width":  data["sepal_width"],
        "petal_length": data["petal_length"],
        "petal_width":  data["petal_width"]
    }])
    prediction = model.predict(sample)
    return jsonify({"prediction": prediction[0]})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
