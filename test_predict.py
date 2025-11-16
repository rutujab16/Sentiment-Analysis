# test_predict.py
import requests, json

API = "http://127.0.0.1:5000/predict"
models = ['logreg','nb','rf']
tests = [
    "I loved the movie it was fantastic and touching",
    "It was okay, not bad but not great",
    "very baddd",
    "this is terrible and I hated it",
    "absolutely brilliant performance"
]

for m in models:
    print("\n--- MODEL:", m, "---")
    for s in tests:
        r = requests.post(API, json={"text": s, "model": m}, timeout=10)
        try:
            print(s, "->", json.dumps(r.json(), ensure_ascii=False))
        except Exception as e:
            print("Error calling API:", e, r.text)
