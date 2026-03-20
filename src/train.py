import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BUCKET_NAME = "21f3000413-mlops-week1-bucket"
data_path = f"gs://{BUCKET_NAME}/data/iris.csv"

mlflow.set_experiment("iris_classification")

hyperparams = [
    {"n_estimators": 50,  "max_depth": 3, "random_state": 42},
    {"n_estimators": 100, "max_depth": 5, "random_state": 42},
    {"n_estimators": 200, "max_depth": 10, "random_state": 42},
]

data = pd.read_csv(data_path)
X = data.drop("species", axis=1)
y = data["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_accuracy = 0
best_run_id = None

for params in hyperparams:
    with mlflow.start_run():
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec  = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1   = f1_score(y_test, preds, average="weighted", zero_division=0)
        mlflow.log_params(params)
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("f1_score",  f1)
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="iris_rf_model")
        print(f"Params: {params} | Accuracy: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_run_id = mlflow.active_run().info.run_id

print(f"Best Run ID : {best_run_id}")
print(f"Best Accuracy: {best_accuracy:.4f}")
