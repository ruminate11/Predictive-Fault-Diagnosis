from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# load trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    current = float(request.form["current"])
    vibration = float(request.form["vibration"])
    temperature = float(request.form["temperature"])

    values = np.array([[current, vibration, temperature]])

    prediction = model.predict(values)[0]

    result = "⚠ Fault Detected" if prediction == 1 else "✅ Normal Condition"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
