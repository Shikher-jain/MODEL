import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('cnn_lstm_model.h5')

def load_data(ndvi_path, sensor_path):
    """
    Load NDVI and sensor data from .npy files.
    Modify this if your data is in another format.
    """
    ndvi_data = np.load(ndvi_path)  # e.g. shape (N, 1, 315, 316, 1)
    sensor_data = np.load(sensor_path)  # e.g. shape (N, 1, 315, 316, 5)
    return ndvi_data, sensor_data

def predict_and_print(ndvi_data, sensor_data):
    preds = model.predict([ndvi_data, sensor_data])
    print("Predictions:")
    for i, pred in enumerate(preds):
        print(f"Sample {i}: {pred}")

if __name__ == "__main__":
    ndvi_path = 'path_to_your_ndvi_data.npy'       # Update with your actual file path
    sensor_path = 'path_to_your_sensor_data.npy'   # Update with your actual file path

    ndvi_data, sensor_data = load_data(ndvi_path, sensor_path)
    predict_and_print(ndvi_data, sensor_data)
