import mlflow.sklearn
import pandas as pd

model = mlflow.sklearn.load_model("models:/iris_rf_model/latest")

sample = pd.DataFrame([{
    "sepal_length": 5.1,
    "sepal_width":  3.5,
    "petal_length": 1.4,
    "petal_width":  0.2
}])

prediction = model.predict(sample)
print(f"Prediction: {prediction[0]}")
