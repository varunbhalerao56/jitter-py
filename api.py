"""
Stress Detection REST API
==========================
Updated to match model_training.ipynb (Mandal et al. 2023 method):
  - 6 features only: l_g_mean_ratio, l_g_median_ratio, l_g_mean_diff,
                     l_g_median_diff, l_diff, l_ssd
  - Chunk size of 40 (paper's optimal)
  - Baseline only needs mean + median (max/min/std not used in any feature)
  - /calibrate endpoint computes baseline automatically from resting HR
  - /predict_user endpoint uses stored baseline by user_id

Endpoints:
  POST /predict       — predict stress (manual baseline)
  POST /predict_user  — predict stress (stored baseline by user_id)
  POST /calibrate     — compute + store baseline from resting HR readings
  GET  /health        — health check

Usage from Swift/Apple Watch:
  Step 1 — calibrate once:
  POST /calibrate
  {
    "user_id": "watch_abc123",
    "resting_hr": [75, 76, 77, 76, 78, 77, 76, 75, 77, 78, ...]
  }

  Step 2 — predict:
  POST /predict_user
  {
    "user_id": "watch_abc123",
    "hr_readings": [90, 92, 95, 98, 101, ...]
  }

Run:
  python api.py
  → Starts on http://0.0.0.0:5050
"""

# ── Prediction threshold ──────────────────────────────────────────────────────
# Raise this to reduce false positives (default 0.50, recommended 0.60-0.70)
STRESS_THRESHOLD = 0.70

from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH    = "stress_model_combined.pkl"
FEATURES_PATH = "feature_names_combined.pkl"

if os.path.exists(MODEL_PATH):
    model         = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Features expected: {feature_names}")
else:
    model         = None
    feature_names = None
    print(f"WARNING: {MODEL_PATH} not found. Run model_training.ipynb first.")

# ── In-memory baseline store (per user/device) ────────────────────────────────
baselines_store = {}


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
        "model_loaded": model is not None,
        "features": feature_names,
        "calibrated_users": list(baselines_store.keys()),
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Manual baseline version — you supply mean and median yourself.

    POST /predict
    {
        "hr_readings": [90, 92, 95, ...],   (at least 40 recommended)
        "baseline": {
            "mean": 77.0,
            "median": 77.0
        }
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run model_training.ipynb first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Validate hr_readings
    hr_readings = data.get('hr_readings')
    if not hr_readings or not isinstance(hr_readings, list) or len(hr_readings) < 5:
        return jsonify({"error": "hr_readings must be a list of at least 5 BPM values"}), 400

    # Validate baseline — only mean and median are required now
    baseline = data.get('baseline')
    if not baseline:
        return jsonify({"error": "baseline is required with keys: mean, median"}), 400

    required_keys = ['mean', 'median']
    missing = [k for k in required_keys if k not in baseline]
    if missing:
        return jsonify({"error": f"baseline missing keys: {missing}"}), 400

    # Use last 40 readings (paper's optimal chunk size)
    chunk = hr_readings[-40:] if len(hr_readings) >= 40 else hr_readings

    try:
        feats = compute_features(chunk, baseline)
        vec   = np.array([[feats[f] for f in feature_names]])

        pred  = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]

        return jsonify({
            "prediction": "STRESSED" if float(probs[1]) >= STRESS_THRESHOLD else "NOT STRESSED",
            "stress_probability": round(float(probs[1]), 4),
            "confidence":         round(float(max(probs)), 4),
            "hr_readings_used":   len(chunk),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/calibrate', methods=['POST'])
def calibrate():
    """
    Accepts resting HR readings, computes baseline, stores it by user_id.
    Call this once when the user first sets up the app (sit still ~1 min).

    POST /calibrate
    {
        "user_id": "watch_abc123",
        "resting_hr": [75, 76, 77, 76, 78, 77, ...]   (at least 40 readings)
    }

    Response:
    {
        "status": "calibrated",
        "user_id": "watch_abc123",
        "baseline": {"mean": 76.8, "median": 77.0},
        "readings_used": 58
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    resting_hr = data.get('resting_hr')
    if not resting_hr or not isinstance(resting_hr, list):
        return jsonify({"error": "resting_hr must be a list of BPM values"}), 400

    # Need at least 40 readings (~40 seconds at 1Hz) for a reliable baseline
    if len(resting_hr) < 40:
        return jsonify({
            "error": f"Need at least 40 resting HR readings, got {len(resting_hr)}. "
                     "Ask the user to sit still for ~1 minute."
        }), 400

    hr = np.array(resting_hr, dtype=float)

    # Remove outliers — readings outside 2 std devs from mean
    mean_val, std_val = np.mean(hr), np.std(hr)
    clean_hr = hr[np.abs(hr - mean_val) <= 2 * std_val]
    if len(clean_hr) < 10:
        clean_hr = hr  # fallback if too many removed

    baseline = {
        "mean":   round(float(np.mean(clean_hr)),   2),
        "median": round(float(np.median(clean_hr)), 2),
    }

    # Store baseline keyed by user_id
    baselines_store[user_id] = baseline

    return jsonify({
        "status":           "calibrated",
        "user_id":          user_id,
        "baseline":         baseline,
        "readings_used":    len(clean_hr),
        "outliers_removed": len(hr) - len(clean_hr),
    })


@app.route('/predict_user', methods=['POST'])
def predict_user():
    """
    Predict using a stored baseline — no need to send baseline each time.
    Requires /calibrate to have been called first for this user_id.

    POST /predict_user
    {
        "user_id": "watch_abc123",
        "hr_readings": [90, 92, 95, 98, 101, ...]
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run model_training.ipynb first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    if user_id not in baselines_store:
        return jsonify({
            "error": f"No baseline found for user '{user_id}'. "
                     "Call /calibrate first with resting HR readings."
        }), 400

    hr_readings = data.get('hr_readings')
    if not hr_readings or not isinstance(hr_readings, list) or len(hr_readings) < 5:
        return jsonify({"error": "hr_readings must be a list of at least 5 BPM values"}), 400

    baseline = baselines_store[user_id]
    chunk    = hr_readings[-40:] if len(hr_readings) >= 40 else hr_readings

    try:
        feats = compute_features(chunk, baseline)
        vec   = np.array([[feats[f] for f in feature_names]])

        pred  = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]

        return jsonify({
            "prediction": "STRESSED" if float(probs[1]) >= STRESS_THRESHOLD else "NOT STRESSED",
            "stress_probability": round(float(probs[1]), 4),
            "confidence":         round(float(max(probs)), 4),
            "hr_readings_used":   len(chunk),
            "baseline_used":      baseline,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)