from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Paths to model and scaler
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Lazy-load model and scaler (only when needed)
model = None
scaler = None

def get_model_and_scaler():
    global model, scaler
    if model is None:
        model = joblib.load(MODEL_PATH)
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)
    return model, scaler

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Capture inputs as a dictionary
    inputs = {
        "GHI": request.form["GHI"],
        "temp": request.form["temp"],
        "pressure": request.form["pressure"],
        "humidity": request.form["humidity"],
        "wind_speed": request.form["wind_speed"],
        "rain_1h": request.form["rain_1h"],
        "snow_1h": request.form["snow_1h"],
        "clouds_all": request.form["clouds_all"],
        "sunlightTime": request.form["sunlightTime"],
        "dayLength": request.form["dayLength"],
        "hour": request.form["hour"],
        "month": request.form["month"],
        "year": request.form["year"]
    }

    # Convert inputs to float for prediction
    features = [float(v) for v in inputs.values()]
    features_array = np.array([features])  # Shape (1, n_features)

    # Get model & scaler
    model, scaler = get_model_and_scaler()
    scaled_features = scaler.transform(features_array)

    # Make prediction
    prediction = model.predict(scaled_features)[0]
    prediction = round(float(prediction), 2)

    # Send both prediction and inputs to result page
    return render_template('result.html', prediction=prediction, inputs=inputs)

# Flask entry point (ignored on PythonAnywhere WSGI)
if __name__ == '__main__':
    app.run(debug=True)
