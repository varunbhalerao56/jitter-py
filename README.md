# Stress Detection API

Predicts stress from heart rate readings using a trained ML model.

## Deploy to Render (Free)

1. Push this folder to a **GitHub repo**
2. Go to [render.com](https://render.com) → Sign up (free, no credit card)
3. Click **New → Web Service**
4. Connect your GitHub repo
5. Render auto-detects the config. Set:
   - **Build Command:** `pip install -r requirements.txt`  
   - **Start Command:** `gunicorn api:app --bind 0.0.0.0:$PORT`
6. Click **Deploy**
7. You'll get a URL like: `https://stress-api-xxxx.onrender.com`

> **Note:** Free tier sleeps after 15 min of inactivity. 
> First request after sleep takes ~30 seconds to wake up.

## API Usage

### Health Check
```
GET /health
```

### Predict Stress
```
POST /predict
Content-Type: application/json

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
```

### Response
```json
{
  "prediction": "STRESSED",
  "stress_probability": 0.87,
  "confidence": 0.87,
  "hr_readings_used": 15
}
```

## Swift Example
```swift
func predictStress(hrReadings: [Double], baseline: [String: Double]) async throws -> StressResult {
    let url = URL(string: "https://your-app.onrender.com/predict")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    
    let body: [String: Any] = [
        "hr_readings": hrReadings,
        "baseline": baseline
    ]
    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    
    let (data, _) = try await URLSession.shared.data(for: request)
    return try JSONDecoder().decode(StressResult.self, from: data)
}
```

## Files
- `api.py` — Flask API server
- `stress_model_combined.pkl` — Trained model (WESAD + SWELL)
- `feature_names_combined.pkl` — Feature order for the model
- `requirements.txt` — Python dependencies
- `render.yaml` — Render deployment config
