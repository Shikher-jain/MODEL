import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_input(ndvi_seq, sensor_seq):
    """
    Apply same normalization/scaling as training.
    """
    ndvi_seq = ndvi_seq / 255.0
    N, T, H, W, C = sensor_seq.shape
    sensor_reshaped = sensor_seq.reshape((N * T * H * W, C))
    scaler = StandardScaler()
    sensor_scaled = scaler.fit_transform(sensor_reshaped)
    sensor_scaled = sensor_scaled.reshape((N, T, H, W, C))
    return ndvi_seq, sensor_scaled
