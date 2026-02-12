from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(
    __name__,
    template_folder="frontend",
    static_folder="Style",
    static_url_path="/static"
)

# Load ML model
model = joblib.load("healthcare_model.pkl")

# Load CSV
data = pd.read_csv("data/patients.csv")

def get_risk_stats():
    high = int((data["risk"] == 1).sum())
    low = int((data["risk"] == 0).sum())
    return high, low

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        age = int(request.form["age"])
        heart_rate = int(request.form["heart_rate"])
        blood_pressure = int(request.form["blood_pressure"])

        features = np.array([[age, heart_rate, blood_pressure]])
        prediction = model.predict(features)[0]

        result = "High Risk" if prediction == 1 else "Low Risk"

    high, low = get_risk_stats()

    return render_template(
        "index.html",
        result=result,
        high=high,
        low=low
    )

if __name__ == "__main__":
    app.run(debug=True)
