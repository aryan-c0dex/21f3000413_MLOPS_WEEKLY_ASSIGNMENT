import pandas as pd

def test_dataset_schema():
    df = pd.read_csv("data/iris.csv")

    expected_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]

    for col in expected_columns:
        assert col in df.columns


def test_no_missing_values():
    df = pd.read_csv("data/iris.csv")
    assert df.isnull().sum().sum() == 0


def test_feature_ranges():
    df = pd.read_csv("data/iris.csv")

    assert df["sepal_length"].between(4, 8).all()
    assert df["sepal_width"].between(2, 5).all()
    assert df["petal_length"].between(1, 7).all()
    assert df["petal_width"].between(0, 3).all()
