import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# --- Config ---
target_height = 315
target_width = 316
ndvi_fp = "data/ndvi/Bathinda_3_2018_ndvi_heatmap.npy"
sensor_fp = "data/sensor/Bathinda_13_2018_Sensor.npy"
model_path = "model.h5"

# --- Functions ---


def load_and_prepare(filepath, expected_channels):
    data = np.load(filepath)

    # Handle structured array with named bands
    if data.dtype.names:
        data = np.stack([data[name] for name in data.dtype.names], axis=-1)

    data = data.astype(np.float32)

    # If 2D, add channel dim
    if data.ndim == 2:
        data = data[..., np.newaxis]

    # Resize each channel individually
    resized_channels = []
    for c in range(data.shape[-1]):
        resized_channel = cv2.resize(data[..., c], (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_channels.append(resized_channel)

    resized = np.stack(resized_channels, axis=-1)

    # ✅ Ensure it's 3D (H, W, C)
    if resized.ndim > 3:
        resized = resized.reshape((target_height, target_width, -1))

    # ✅ Pad channels if fewer than expected
    current_channels = resized.shape[-1]
    if current_channels < expected_channels:
        padding = np.zeros((target_height, target_width, expected_channels - current_channels), dtype=np.float32)
        resized = np.concatenate([resized, padding], axis=-1)

    return resized


def preprocess(ndvi_data, sensor_data):
    # Normalize NDVI
    ndvi_data = ndvi_data / 255.0

    # Standardize sensor channels
    reshaped = sensor_data.reshape(-1, sensor_data.shape[-1])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(reshaped)
    sensor_data = scaled.reshape(sensor_data.shape)

    return ndvi_data, sensor_data

def predict_yield():
    print("Loading model...")
    model = load_model(model_path, compile=False)

    print("Loading and processing NDVI data...")
    ndvi = load_and_prepare(ndvi_fp, expected_channels=1)

    print("Loading and processing sensor data...")
    sensor = load_and_prepare(sensor_fp, expected_channels=5)

    # Add time step = 1 and batch = 1
    ndvi = np.expand_dims(ndvi, axis=0)  # batch
    ndvi = np.expand_dims(ndvi, axis=1)  # time
    sensor = np.expand_dims(sensor, axis=0)
    sensor = np.expand_dims(sensor, axis=1)

    ndvi, sensor = preprocess(ndvi, sensor)

    # Predict
    pred = model.predict([ndvi, sensor])
    print(f"\n🌾 Predicted yield: {pred[0][0]:.2f}")

# --- Run ---
if __name__ == "__main__":
    predict_yield()
