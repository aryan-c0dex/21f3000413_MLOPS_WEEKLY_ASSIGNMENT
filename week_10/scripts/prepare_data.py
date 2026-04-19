import json
import random
from sklearn.datasets import load_iris

CLASS_NAMES = ["setosa", "versicolor", "virginica"]
random.seed(42)

iris = load_iris()
samples = []
for feat, label in zip(iris.data, iris.target):
    samples.append({
        "sepal_length": round(float(feat[0]), 1),
        "sepal_width":  round(float(feat[1]), 1),
        "petal_length": round(float(feat[2]), 1),
        "petal_width":  round(float(feat[3]), 1),
        "species":      CLASS_NAMES[label],
    })

random.shuffle(samples)
split = int(len(samples) * 0.8)
train = samples[:split]
test  = samples[split:]

def to_v1(s):
    return {
        "input_text":  f"sepal_length: {s['sepal_length']}, sepal_width: {s['sepal_width']}, "
                       f"petal_length: {s['petal_length']}, petal_width: {s['petal_width']}",
        "output_text": s["species"],
    }

def to_v2(s):
    return {
        "input_text":  f"A flower specimen has a sepal length of {s['sepal_length']} cm, "
                       f"sepal width of {s['sepal_width']} cm, "
                       f"petal length of {s['petal_length']} cm, "
                       f"and petal width of {s['petal_width']} cm. "
                       f"Identify the iris species.",
        "output_text": f"This is Iris {s['species']}.",
    }

def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} records -> {path}")

write_jsonl([to_v1(s) for s in train], "week_10/data/iris_v1_train.jsonl")
write_jsonl([to_v1(s) for s in test],  "week_10/data/iris_v1_test.jsonl")
write_jsonl([to_v2(s) for s in train], "week_10/data/iris_v2_train.jsonl")
write_jsonl([to_v2(s) for s in test],  "week_10/data/iris_v2_test.jsonl")

print("\n-- Sample v1 record --")
print(json.dumps(to_v1(samples[0]), indent=2))

print("\n-- Sample v2 record --")
print(json.dumps(to_v2(samples[0]), indent=2))

print("\nDone! 4 files created in week_10/data/")
