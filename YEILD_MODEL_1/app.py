# app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import joblib
import tempfile
from utils.preprocess import preprocess_input
from typing import Tuple


app = FastAPI(title="Crop Yield Prediction API")

# --- Load model and scaler with error handling ---
import logging
model = None
scaler = None
model_error = None
scaler_error = None
try:
    model = tf.keras.models.load_model("model.h5", compile=False)
except Exception as e:
    model_error = str(e)
    logging.warning(f"Error loading model: {e}")
try:
    scaler = joblib.load("scaler.save")
except Exception as e:
    scaler_error = str(e)
    logging.warning(f"Error loading scaler: {e}")
@app.get("/health")
def health():
    status = {"model_loaded": model is not None, "scaler_loaded": scaler is not None}
    if model_error:
        status["model_error"] = model_error
    if scaler_error:
        status["scaler_error"] = scaler_error
    return status

@app.get("/")
def root():
    return {"message": "🌾 Crop Yield API is up! Upload NDVI and Sensor files."}


@app.post("/predict")
async def predict(
    ndvi_file: UploadFile = File(..., description="NDVI heatmap .npy file"),
    sensor_file: UploadFile = File(..., description="Sensor .npy file (5 channels)")
):
    # --- Check model and scaler loaded ---
    if model is None or scaler is None:
        msg = "Model or scaler not loaded. "
        if model_error:
            msg += f"Model error: {model_error}. "
        if scaler_error:
            msg += f"Scaler error: {scaler_error}. "
        return JSONResponse(status_code=500, content={"error": msg.strip()})

    # --- Validate file types ---
    if not ndvi_file.filename.endswith('.npy') or not sensor_file.filename.endswith('.npy'):
        return JSONResponse(status_code=400, content={"error": "Both files must be .npy format."})

    try:
        # --- Save files temporarily ---
        with tempfile.TemporaryDirectory() as tmpdir:
            ndvi_path = f"{tmpdir}/ndvi.npy"
            sensor_path = f"{tmpdir}/sensor.npy"

            with open(ndvi_path, "wb") as f:
                f.write(await ndvi_file.read())
            with open(sensor_path, "wb") as f:
                f.write(await sensor_file.read())

            # --- Load files ---
            ndvi = np.load(ndvi_path)
            sensor = np.load(sensor_path)

            # --- Validate array shapes ---
            if ndvi.ndim not in [2, 3]:
                return JSONResponse(status_code=400, content={"error": f"NDVI array must be 2D or 3D, got shape {ndvi.shape}"})
            if sensor.ndim not in [2, 3]:
                return JSONResponse(status_code=400, content={"error": f"Sensor array must be 2D or 3D, got shape {sensor.shape}"})

            # --- Ensure channel dims ---
            if ndvi.ndim == 2:
                ndvi = ndvi[..., np.newaxis]  # (H, W, 1)
            if sensor.ndim == 2:
                sensor = sensor[..., np.newaxis]  # (H, W, 1)

            # --- Add time & batch dims ---
            ndvi = np.expand_dims(ndvi, axis=0)   # (1, H, W, C)
            ndvi = np.expand_dims(ndvi, axis=1)   # (1, 1, H, W, C)
            sensor = np.expand_dims(sensor, axis=0)
            sensor = np.expand_dims(sensor, axis=1)

            # --- Preprocess ---
            ndvi, sensor = preprocess_input(ndvi, sensor, scaler)

            # --- Predict ---
            prediction = model.predict([ndvi, sensor])[0][0]
            return {"predicted_yield": round(float(prediction), 2)}

    except Exception as e:
        import traceback
        logging.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})
