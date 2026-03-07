from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32
from feast.value_type import ValueType
from datetime import timedelta

# 1️⃣ Data Source (COURSE-PROVIDED CSV)
iris_source = FileSource(
    path="../../../data/iris_data_adapted_for_feast.parquet",
    event_timestamp_column="event_timestamp",
)

# 2️⃣ Entity
iris = Entity(
    name="iris_id",
    value_type=ValueType.INT64,
    join_keys=["iris_id"],
)

# 3️⃣ Feature View
iris_features = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    online=True,
    source=iris_source,
)
