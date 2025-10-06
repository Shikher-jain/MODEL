
from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
import tempfile, zipfile, os
from utils.preprocess import preprocess_input

app = FastAPI(title="Crop Yield Prediction API")

MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Crop Yield Prediction API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a .zip file containing:
      - NDVI .npy sequence
      - sensor .npy sequence
    Returns yield or crop health prediction.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "input.zip")
        with open(zip_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract uploaded zip
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        ndvi_files = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if "ndvi" in f and f.endswith(".npy")])
        sensor_files = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if "sensor" in f and f.endswith(".npy")])

        if not ndvi_files or not sensor_files:
            return {"error": "Missing NDVI or sensor .npy files in zip."}

        # Load and reshape sequences
        time_steps = len(ndvi_files)
        ndvi_seq = np.stack([np.load(f) for f in ndvi_files], axis=0)
        sensor_seq = np.stack([np.load(f) for f in sensor_files], axis=0)

        # Ensure proper dimensions
        if ndvi_seq.ndim == 3:
            ndvi_seq = ndvi_seq[..., np.newaxis]
        if sensor_seq.ndim == 3:
            sensor_seq = sensor_seq[..., np.newaxis]

        ndvi_seq = np.expand_dims(ndvi_seq, axis=0)      # (1, T, H, W, 1)
        sensor_seq = np.expand_dims(sensor_seq, axis=0)  # (1, T, H, W, 5)

        ndvi_seq, sensor_seq = preprocess_input(ndvi_seq, sensor_seq)

        # Prediction
        prediction = model.predict([ndvi_seq, sensor_seq])[0][0]
        return {"predicted_yield": float(prediction)}


'''
uvicorn app:app --reload
'''