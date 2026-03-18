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
# ── Feature computation — Mandal et al. (2023), 6 features exactly ───────────
def compute_features(hr_chunk, global_stats):
    """
    Exactly 6 features from the paper (Equations 1-6).
    Only global mean and global median are needed — max/min/std are not
    used in any of the 6 features.

    l_g_mean_ratio   = local mean   / global mean         (Eq. 1)
    l_g_median_ratio = local median / global median       (Eq. 2)
    l_g_mean_diff    = sqrt((local mean   - global mean)²)   (Eq. 3)
    l_g_median_diff  = sqrt((local median - global median)²) (Eq. 4)
    l_diff           = local max HR - local min HR        (Eq. 5)
    l_ssd            = std dev of HR values in chunk      (Eq. 6)
    """
    hr = np.array(hr_chunk, dtype=float)
    g  = global_stats

    local_mean   = np.mean(hr)
    local_median = np.median(hr)

    return {
        'l_g_mean_ratio':   local_mean   / g['mean']   if g['mean']   != 0 else 1.0,
        'l_g_median_ratio': local_median / g['median'] if g['median'] != 0 else 1.0,
        'l_g_mean_diff':    np.sqrt((local_mean   - g['mean'])   ** 2),
        'l_g_median_diff':  np.sqrt((local_median - g['median']) ** 2),
        'l_diff':           float(np.max(hr) - np.min(hr)),
        'l_ssd':            float(np.std(hr)),
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
