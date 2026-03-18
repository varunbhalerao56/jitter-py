# Anxiety Detection API

Predicts anxiety from heart rate readings using a trained ML model.

## Usage

```bash
python api.py
```

## Endpoints

### `GET /health`

Health check endpoint.

### `POST /predict`

Predict anxiety from heart rate readings.

Request body:

```json
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

Response:

```json
{
  "prediction": "ANXIOUS",
  "stress_probability": 0.87,
  "confidence": 0.87,
  "hr_readings_used": 15
}
```

