# app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import joblib
import tempfile
from utils.preprocess import preprocess_input

app = FastAPI(title="Crop Yield Prediction API (No ZIP)")

# Load model and scaler
model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.save")

@app.get("/")
def root():
    return {"message": "🌾 Crop Yield API is up! Upload NDVI and Sensor files."}


@app.post("/predict")
async def predict(
    ndvi_file: UploadFile = File(..., description="NDVI heatmap .npy file"),
    sensor_file: UploadFile = File(..., description="Sensor .npy file (5 channels)")
):
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
        return JSONResponse(status_code=500, content={"error": str(e)})

'''
uvicorn app:app --reload
'''