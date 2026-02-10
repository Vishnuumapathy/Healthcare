import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load ML model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "healthcare_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return "Healthcare ML API running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Basic validation
    required_fields = ["age", "heart_rate", "blood_pressure"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    features = np.array([[
        data["age"],
        data["heart_rate"],
        data["blood_pressure"]
    ]])

    prediction = model.predict(features)[0]
    result = "High Risk" if prediction == 1 else "Low Risk"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
