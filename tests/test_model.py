import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_accuracy():

    model = joblib.load("model/model.pkl")

    df = pd.read_csv("data/iris.csv")

    X = df.drop(columns=["species"])
    y = df["species"]

    preds = model.predict(X)

    acc = accuracy_score(y, preds)

    assert acc > 0.7
