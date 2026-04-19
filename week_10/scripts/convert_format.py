import json

def convert(input_path, output_path):
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            rec = json.loads(line.strip())
            new_rec = {
                "contents": [
                    {"role": "user",  "parts": [{"text": rec["input_text"]}]},
                    {"role": "model", "parts": [{"text": rec["output_text"]}]}
                ]
            }
            f_out.write(json.dumps(new_rec) + "\n")
    print(f"Converted {input_path} -> {output_path}")

convert("week_10/data/iris_v1_train.jsonl", "week_10/data/iris_v1_train_gemini.jsonl")
convert("week_10/data/iris_v1_test.jsonl",  "week_10/data/iris_v1_test_gemini.jsonl")
convert("week_10/data/iris_v2_train.jsonl", "week_10/data/iris_v2_train_gemini.jsonl")
convert("week_10/data/iris_v2_test.jsonl",  "week_10/data/iris_v2_test_gemini.jsonl")
