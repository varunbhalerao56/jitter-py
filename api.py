"""
Stress Detection REST API
==========================
Simple Flask API that takes heart rate readings + personal baseline
and returns stress prediction.

Endpoints:
  POST /predict  — predict stress from HR readings
  GET  /health   — health check

Usage from Swift/Apple Watch:
  POST /predict
  {
    "hr_readings": [72, 75, 78, 82, 85, 88, 90, 92, 95, 93, 91, 94, 96, 93, 90],
    "baseline": {
      "mean": 72.0,
      "median": 71.5,
      "max": 95.0,
      "min": 58.0,
      "std": 6.2
    }
  }

Response:
  {
    "prediction": "STRESSED",
    "stress_probability": 0.87,
    "confidence": 0.87
  }

Run:
  python api.py
  → Starts on http://0.0.0.0:5050
"""

from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = "stress_model_combined.pkl"
FEATURES_PATH = "feature_names_combined.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print(f"Model loaded: {MODEL_PATH}")
else:
    model = None
    feature_names = None
    print(f"WARNING: {MODEL_PATH} not found. Run stress_combined.py first to train.")


# ── Feature computation (same as training) ────────────────────────────────────
def compute_features(hr_chunk, global_stats):
    hr = np.array(hr_chunk, dtype=float)

    local_mean = np.mean(hr)
    local_median = np.median(hr)
    local_max = np.max(hr)
    local_min = np.min(hr)
    local_std = np.std(hr)
    local_range = local_max - local_min

    g = global_stats

    mean_ratio = local_mean / g['mean'] if g['mean'] != 0 else 1.0
    median_ratio = local_median / g['median'] if g['median'] != 0 else 1.0
    max_ratio = local_max / g['max'] if g['max'] != 0 else 1.0
    min_ratio = local_min / g['min'] if g['min'] != 0 else 1.0
    std_ratio = local_std / g['std'] if g['std'] != 0 else 1.0

    mean_diff = np.sqrt((local_mean - g['mean'])**2)
    median_diff = np.sqrt((local_median - g['median'])**2)
    max_diff = np.sqrt((local_max - g['max'])**2)
    min_diff = np.sqrt((local_min - g['min'])**2)

    if len(hr) >= 4:
        half = len(hr) // 2
        trend = np.mean(hr[half:]) - np.mean(hr[:half])
    else:
        trend = 0.0

    cv = local_std / local_mean if local_mean != 0 else 0.0

    return {
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'max_ratio': max_ratio,
        'min_ratio': min_ratio,
        'std_ratio': std_ratio,
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'max_diff': max_diff,
        'min_diff': min_diff,
        'local_range': local_range,
        'local_std': local_std,
        'trend': trend,
        'cv': cv,
        'local_mean': local_mean,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Train first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Validate hr_readings
    hr_readings = data.get('hr_readings')
    if not hr_readings or not isinstance(hr_readings, list) or len(hr_readings) < 5:
        return jsonify({"error": "hr_readings must be a list of at least 5 BPM values"}), 400

    # Validate baseline
    baseline = data.get('baseline')
    if not baseline:
        return jsonify({"error": "baseline is required with keys: mean, median, max, min, std"}), 400

    required_keys = ['mean', 'median', 'max', 'min', 'std']
    missing = [k for k in required_keys if k not in baseline]
    if missing:
        return jsonify({"error": f"baseline missing keys: {missing}"}), 400

    # Use last 15 readings (or all if fewer)
    chunk = hr_readings[-15:] if len(hr_readings) >= 15 else hr_readings

    try:
        feats = compute_features(chunk, baseline)
        vec = np.array([[feats[f] for f in feature_names]])

        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]

        return jsonify({
            "prediction": "STRESSED" if pred == 1 else "NOT STRESSED",
            "stress_probability": round(float(probs[1]), 4),
            "confidence": round(float(max(probs)), 4),
            "hr_readings_used": len(chunk),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)