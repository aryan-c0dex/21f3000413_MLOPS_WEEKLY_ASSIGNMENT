import re
import json
from datetime import datetime

BLOCKLIST = [
    "ignore previous instructions",
    "ignore all instructions",
    "you are now",
    "dan mode",
    "forget flower classification",
    "system prompt",
    "context window",
    "training examples",
    "repeat everything",
    "what instructions were you given",
    "translate",
    "what is 2+2",
    "capital of france",
]

IRIS_SCHEMA_PATTERN = re.compile(
    r"sepal.length.*sepal.width.*petal.length.*petal.width",
    re.IGNORECASE
)

VALID_SPECIES = ["setosa", "versicolor", "virginica"]

audit_log = []

def input_guardrail(prompt):
    timestamp = datetime.utcnow().isoformat()
    prompt_lower = prompt.lower()

    # Check 1 — blocklist
    for keyword in BLOCKLIST:
        if keyword in prompt_lower:
            entry = {"timestamp": timestamp, "rule": "blocklist", "matched": keyword, "input": prompt[:100]}
            audit_log.append(entry)
            print(f"[BLOCKED] Rule=blocklist | Matched='{keyword}'")
            return {"blocked": True, "reason": f"Blocked keyword: {keyword}"}

    # Check 2 — schema validation
    if not IRIS_SCHEMA_PATTERN.search(prompt):
        entry = {"timestamp": timestamp, "rule": "schema_check", "matched": "invalid_schema", "input": prompt[:100]}
        audit_log.append(entry)
        print(f"[BLOCKED] Rule=schema_check | Input doesn't match IRIS feature format")
        return {"blocked": True, "reason": "Input does not match expected IRIS feature schema"}

    return {"blocked": False}

def output_guardrail(response):
    response_lower = response.lower()

    # Check 1 — format violation
    if not any(v in response_lower for v in VALID_SPECIES):
        return {"filtered": True, "reason": "format_violation", "safe_response": "Unable to classify: invalid output format"}

    # Check 2 — context leakage
    leakage_patterns = ["system prompt", "trained by google", "large language model", "training example"]
    for pattern in leakage_patterns:
        if pattern in response_lower:
            return {"filtered": True, "reason": "context_leakage", "safe_response": "Unable to classify: response contains sensitive context"}

    return {"filtered": False, "safe_response": response}

def save_audit():
    with open("week_11/audit_log.json", "w") as f:
        json.dump(audit_log, f, indent=2)
    print(f"Audit log saved: {len(audit_log)} entries")
