from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load model once at startup
model = joblib.load("healthcare_model.pkl")

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Healthcare Risk Prediction</title>
</head>
<body>
    <h2>Healthcare Risk Prediction</h2>

    <form method="post">
        Age: <input type="number" name="age" required><br><br>
        Heart Rate: <input type="number" name="heart_rate" required><br><br>
        Blood Pressure: <input type="number" name="blood_pressure" required><br><br>
        <button type="submit">Predict</button>
    </form>

    {% if result %}
        <h3>Prediction: {{ result }}</h3>
    {% endif %}
</body>
</html>
"""

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

    return render_template_string(HTML_PAGE, result=result)

if __name__ == "__main__":
    app.run(debug=True)
