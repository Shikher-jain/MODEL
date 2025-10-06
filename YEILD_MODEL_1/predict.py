# predict.py

import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
import os

# --- Config ---
target_height = 315
target_width = 316
ndvi_fp = "data/ndvi/Bathinda_3_2018_ndvi_heatmap.npy"
sensor_fp = "data/sensor/Bathinda_13_2018_Sensor.npy"
model_path = "model.h5"
scaler_path = "scaler.save"

# --- Load model and scaler ---
print("🔄 Loading model and scaler...")
model = load_model(model_path, compile=False)
scaler = joblib.load(scaler_path)

# --- Load & preprocess functions ---

def load_and_prepare(filepath, expected_channels):
    data = np.load(filepath)

    # Handle structured array
    if data.dtype.names:
        data = np.stack([data[name] for name in data.dtype.names], axis=-1)

    data = data.astype(np.float32)

    if data.ndim == 2:
        data = data[..., np.newaxis]

    resized_channels = []
    for c in range(data.shape[-1]):
        resized = cv2.resize(data[..., c], (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_channels.append(resized)

    data_resized = np.stack(resized_channels, axis=-1)

    if data_resized.shape[-1] < expected_channels:
        padding = np.zeros((target_height, target_width, expected_channels - data_resized.shape[-1]), dtype=np.float32)
        data_resized = np.concatenate([data_resized, padding], axis=-1)

    return data_resized

def preprocess(ndvi_data, sensor_data, scaler):
    # Normalize NDVI
    ndvi_data = ndvi_data / 255.0

    # Standardize sensor channels
    reshaped = sensor_data.reshape(-1, sensor_data.shape[-1])
    scaled = scaler.transform(reshaped)
    sensor_data = scaled.reshape(sensor_data.shape)

    return ndvi_data, sensor_data

# --- Main prediction ---
def predict_yield():
    print("📥 Loading and processing NDVI...")
    ndvi = load_and_prepare(ndvi_fp, expected_channels=1)

    print("📥 Loading and processing Sensor data...")
    sensor = load_and_prepare(sensor_fp, expected_channels=5)

    # Add batch and time dimensions
    ndvi = np.expand_dims(ndvi, axis=(0, 1))     # (1, 1, H, W, 1)
    sensor = np.expand_dims(sensor, axis=(0, 1)) # (1, 1, H, W, 5)

    # Preprocess
    ndvi, sensor = preprocess(ndvi, sensor, scaler)

    # Predict
    print("🔮 Predicting...")
    pred = model.predict([ndvi, sensor])
    print(f"\n🌾 Predicted yield: {pred[0][0]:.2f}")

# --- Run ---
if __name__ == "__main__":
    predict_yield()

