from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ==============================
# Load trained ML model
# ==============================
model = joblib.load("model.pkl")


# ==============================
# Graph Generator (Compact + Clean)
# ==============================
def create_graph(values, new_val, label, is_fault):
    values = list(values)
    values.append(new_val)

    normal_level = np.mean(values[:-1])

    plt.figure(figsize=(3.8, 2.6))  # slightly bigger for readability

    x = range(len(values))

    # trend line
    plt.plot(
        x,
        values,
        color="#16a34a",   # green
        linewidth=2,
        marker="o",
        label="Trend"
    )

    # normal threshold line
    plt.axhline(
        normal_level,
        color="#22c55e",
        linestyle="--",
        linewidth=1.5,
        label="Normal Level"
    )

    # anomaly / current point
    point_color = "#ef4444" if is_fault else "#16a34a"
    plt.scatter(
        len(values)-1,
        new_val,
        color=point_color,
        s=120,
        zorder=5,
        label="Current Value"
    )

    # titles + labels
    plt.title(label, fontsize=11)
    plt.xlabel("Time")
    plt.ylabel(label)

    # grid for clarity
    plt.grid(True, alpha=0.3)

    plt.legend(fontsize=7)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    buf.seek(0)

    graph = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return graph


# ==============================
# Routes
# ==============================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # get inputs
    current = float(request.form["current"])
    vibration = float(request.form["vibration"])
    temperature = float(request.form["temperature"])

    # model prediction
    values = np.array([[current, vibration, temperature]])
    prediction = model.predict(values)[0]

    is_fault = prediction == 1
    result = "⚠ Fault Detected" if is_fault else "✅ Normal Condition"

    # ------------------------------
    # Generate sample historical data
    # (simulate sensor history)
    # ------------------------------
    current_hist = np.random.normal(current * 0.9, 0.4, 12)
    vib_hist = np.random.normal(vibration * 0.9, 0.25, 12)
    temp_hist = np.random.normal(temperature * 0.95, 0.8, 12)

    # create graphs
    graphs = {
        "current": create_graph(current_hist, current, "Current (A)", is_fault),
        "vibration": create_graph(vib_hist, vibration, "Vibration (mm/s)", is_fault),
        "temperature": create_graph(temp_hist, temperature, "Temperature (°C)", is_fault)
    }

    return render_template(
        "index.html",
        prediction_text=result,
        graphs=graphs
    )


# ==============================
# Run server
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
