import json
import re
import time
import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID  = "mindful-girder-487814-j2"
LOCATION    = "us-central1"

ENDPOINT_V1 = "projects/mindful-girder-487814-j2/locations/us-central1/endpoints/5717091578284081152"
ENDPOINT_V2 = "projects/mindful-girder-487814-j2/locations/us-central1/endpoints/2161499672475074560"

VALID = {"setosa", "versicolor", "virginica"}

vertexai.init(project=PROJECT_ID, location=LOCATION)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def extract_v1(raw):
    c = raw.strip().lower()
    for sp in VALID:
        if sp in c: return sp, True
    return None, False

def extract_v2(raw):
    m = re.search(r"iris\s+(setosa|versicolor|virginica)", raw, re.I)
    if m: return m.group(1).lower(), True
    for sp in VALID:
        if sp in raw.lower(): return sp, False
    return None, False

def evaluate(endpoint, test_path, version):
    print(f"\nEvaluating {version}...")
    model   = GenerativeModel(endpoint)
    records = load_jsonl(test_path)
    correct = 0; compliant = 0; exact = 0

    for i, rec in enumerate(records, 1):
        prompt   = rec["contents"][0]["parts"][0]["text"]
        expected = rec["contents"][1]["parts"][0]["text"]

        try:
            raw = model.generate_content(prompt).text.strip()
        except Exception as e:
            print(f"  Error on record {i}: {e}")
            raw = ""

        if version == "v1":
            pred, comp = extract_v1(raw)
            true_label = expected.strip().lower()
        else:
            pred, comp = extract_v2(raw)
            m = re.search(r"iris\s+(setosa|versicolor|virginica)", expected, re.I)
            true_label = m.group(1).lower() if m else expected.lower()

        is_correct = pred == true_label
        is_exact   = raw.strip() == expected.strip()

        if is_correct: correct   += 1
        if comp:       compliant += 1
        if is_exact:   exact     += 1

        print(f"  [{i}/30] true={true_label} pred={pred} correct={is_correct}")
        time.sleep(1)

    n = len(records)
    metrics = {
        "accuracy":           round(correct/n,   4),
        "exact_match":        round(exact/n,     4),
        "format_compliance":  round(compliant/n, 4),
    }
    print(f"\n── {version} Results ──")
    print(f"  Accuracy          : {metrics['accuracy']:.2%}")
    print(f"  Exact Match       : {metrics['exact_match']:.2%}")
    print(f"  Format Compliance : {metrics['format_compliance']:.2%}")
    return metrics

m1 = evaluate(ENDPOINT_V1, "week_10/data/iris_v1_test_gemini.jsonl", "v1")
m2 = evaluate(ENDPOINT_V2, "week_10/data/iris_v2_test_gemini.jsonl", "v2")

print("\n╔══════════════════════════════════════════════════╗")
print("║           v1 vs v2 Comparison Summary            ║")
print("╠══════════════════════════════════════════════════╣")
print(f"║  Accuracy          v1={m1['accuracy']:.2%}   v2={m2['accuracy']:.2%}      ║")
print(f"║  Exact Match       v1={m1['exact_match']:.2%}   v2={m2['exact_match']:.2%}      ║")
print(f"║  Format Compliance v1={m1['format_compliance']:.2%}   v2={m2['format_compliance']:.2%}      ║")
print("╚══════════════════════════════════════════════════╝")

import os
os.makedirs("week_10/evaluation", exist_ok=True)
with open("week_10/evaluation/results.json", "w") as f:
    json.dump({"v1": m1, "v2": m2}, f, indent=2)
print("\nResults saved to week_10/evaluation/results.json")
