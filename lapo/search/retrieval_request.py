import requests

# URL for your local FastAPI server
url = "http://127.0.0.1:8000/retrieve"

# Example payload
payload = {
    "queries": ["What is the capital of France?", "Explain neural networks."],
    "topk": 2,
    "return_scores": True
}

# Send POST request
response = requests.post(url, json=payload)

# Raise an exception if the request failed
response.raise_for_status()

# Get the JSON response
retrieved_data = response.json()

print("Response from server:")
print(retrieved_data)

import json

data = retrieved_data

queries = payload["queries"]          # ["What is the capital of France?", ...]
topk = payload["topk"]                # 3
results_per_query = data["result"]    # list of list of dicts

output_file = "retrieval_results.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for query, docs in zip(queries, results_per_query):
        line = {
            "query": query,
            "topk": topk,
            "results": docs      # docs 已经包含 id, contents, score
        }
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

print(f"Saved {len(queries)} entries to {output_file}")