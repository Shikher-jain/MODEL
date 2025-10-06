# utils/preprocess.py

import numpy as np

def preprocess_input(ndvi_data, sensor_data, scaler):
    ndvi_data = ndvi_data / 255.0

    N, T, H, W, C = sensor_data.shape
    reshaped = sensor_data.reshape(N * T * H * W, C)
    scaled = scaler.transform(reshaped)
    sensor_data = scaled.reshape(N, T, H, W, C)

    return ndvi_data, sensor_data
