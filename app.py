# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import glob, os, joblib, re, traceback

MODEL_DIR = "saved_models"
PORT = 5000

app = Flask(__name__)
CORS(app)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# load saved models
models = {}
for path in glob.glob(os.path.join(MODEL_DIR, "*_pipeline.joblib")):
    name = os.path.basename(path).split('_pipeline.joblib')[0]
    try:
        bundle = joblib.load(path)
        if isinstance(bundle, dict) and 'pipeline' in bundle:
            models[name] = bundle
        else:
            models[name] = {'pipeline': bundle, 'label_map': None}
        print("Loaded model:", name)
    except Exception as e:
        print("Failed loading", path, e)

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": list(models.keys())})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        text = data.get("text", "")
        model_name = data.get("model", None)
        if not text:
            return jsonify({"error": "No text provided."}), 400
        if not model_name:
            return jsonify({"error": "No model provided."}), 400
        if model_name not in models:
            return jsonify({"error": f"Model '{model_name}' not found.", "available": list(models.keys())}), 400

        bundle = models[model_name]
        pipe = bundle.get('pipeline')
        saved_label_map = bundle.get('label_map')

        cleaned = clean_text(text)
        pred_raw = pipe.predict([cleaned])[0]

        prob_vector = None
        try:
            prob_vector = pipe.predict_proba([cleaned])[0].tolist()
        except Exception:
            prob_vector = None

        # map prediction to human label
        label_str = None
        if saved_label_map:
            inv = {int(v): k for k, v in saved_label_map.items()}
            try:
                pid = int(pred_raw)
                label_str = inv.get(pid, str(pred_raw))
            except Exception:
                label_str = str(pred_raw)
        else:
            clf = None
            try:
                clf = pipe.named_steps.get('clf', None)
            except Exception:
                clf = None
            if clf is not None and hasattr(clf, 'classes_'):
                classes = list(clf.classes_)
                # if pred_raw is in classes (string), use it; else try numeric index
                if pred_raw in classes:
                    label_str = str(pred_raw)
                else:
                    try:
                        pid = int(pred_raw)
                        if 0 <= pid < len(classes):
                            label_str = str(classes[pid])
                        else:
                            label_str = str(pred_raw)
                    except Exception:
                        label_str = str(pred_raw)
            else:
                label_str = str(pred_raw)

        return jsonify({
            "input_text": text,
            "cleaned_text": cleaned,
            "model": model_name,
            "pred_raw": str(pred_raw),
            "label": label_str,
            "probabilities": prob_vector
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

if __name__ == "__main__":
    print("Available models:", list(models.keys()))
    app.run(host="0.0.0.0", port=PORT, debug=True)
