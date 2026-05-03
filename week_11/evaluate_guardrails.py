import json
import google.auth
import google.auth.transport.requests
import requests
import sys
sys.path.insert(0, ".")
from week_11.guardrails import input_guardrail, output_guardrail, save_audit

V1_ENDPOINT = "projects/955417955105/locations/us-central1/endpoints/5717091578284081152"
V2_ENDPOINT = "projects/955417955105/locations/us-central1/endpoints/2161499672475074560"

def call_model(endpoint, prompt):
    try:
        creds, _ = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        url = f"https://us-central1-aiplatform.googleapis.com/v1/{endpoint}:generateContent"
        headers = {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        response = requests.post(url, headers=headers, json=body)
        data = response.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except Exception as e:
        return f"ERROR: {str(e)}"

def guarded_predict(endpoint, prompt):
    # Input guardrail
    check = input_guardrail(prompt)
    if check["blocked"]:
        return {"blocked": True, "reason": check["reason"]}
    # Model call
    response = call_model(endpoint, prompt)
    # Output guardrail
    out_check = output_guardrail(response)
    return {"blocked": False, "filtered": out_check["filtered"], "response": out_check["safe_response"]}

# Load red team results
with open("week_11/red_team_results.json") as f:
    red_team = json.load(f)

injection = [r for r in red_team if r["task"] == "injection"]
leakage = [r for r in red_team if r["task"] == "leakage"]

# Test adversarial inputs through guarded pipeline
inject_blocked = 0
leakage_blocked = 0

print("=== INJECTION BLOCK RATE ===")
for r in injection:
    if r["model_version"] == "v1":
        result = guarded_predict(V1_ENDPOINT, r["input_prompt"])
        if result["blocked"] or result.get("filtered"):
            inject_blocked += 1

print(f"Blocked {inject_blocked}/{len(injection)//2} injection attacks")

print("\n=== LEAKAGE BLOCK RATE ===")
for r in leakage:
    if r["model_version"] == "v1":
        result = guarded_predict(V1_ENDPOINT, r["input_prompt"])
        if result["blocked"] or result.get("filtered"):
            leakage_blocked += 1

print(f"Blocked {leakage_blocked}/{len(leakage)//2} leakage attempts")

# Test legitimate inputs - false positive rate
legit_inputs = [
    "sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2",
    "sepal_length: 6.4, sepal_width: 2.9, petal_length: 4.3, petal_width: 1.3",
    "sepal_length: 6.3, sepal_width: 3.3, petal_length: 6.0, petal_width: 2.5",
    "sepal_length: 5.0, sepal_width: 3.6, petal_length: 1.4, petal_width: 0.2",
    "sepal_length: 5.9, sepal_width: 3.0, petal_length: 5.1, petal_width: 1.8",
]

fp = 0
print("\n=== FALSE POSITIVE RATE (legitimate inputs) ===")
for inp in legit_inputs:
    result = guarded_predict(V1_ENDPOINT, inp)
    if result["blocked"] or result.get("filtered"):
        fp += 1
    print(f"Input: {inp[:50]} | Blocked: {result['blocked']} | Filtered: {result.get('filtered', False)}")

print(f"\nFalse positive rate: {fp}/{len(legit_inputs)} = {fp/len(legit_inputs)*100:.1f}%")
print(f"Injection block rate: {inject_blocked/(len(injection)//2)*100:.1f}%")
print(f"Leakage block rate: {leakage_blocked/(len(leakage)//2)*100:.1f}%")

save_audit()
