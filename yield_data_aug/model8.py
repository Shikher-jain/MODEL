import os
import numpy as np
import pandas as pd
import random
import cv2

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Concatenate, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def augment_sample(ndvi_sample, sensor_sample):
    """
    Apply the same augmentation to both ndvi_sample and sensor_sample.
    Each sample shape: (T, H, W, C)
    """
    # Horizontal flip
    if random.random() < 0.5:
        ndvi_sample = np.flip(ndvi_sample, axis=2)
        sensor_sample = np.flip(sensor_sample, axis=2)
    # Vertical flip
    if random.random() < 0.5:
        ndvi_sample = np.flip(ndvi_sample, axis=1)
        sensor_sample = np.flip(sensor_sample, axis=1)
    # 90-degree rotation (k = 1, 2, 3)
    if random.random() < 0.5:
        k = random.choice([1, 2, 3])
        ndvi_sample = np.rot90(ndvi_sample, k=k, axes=(1, 2))
        sensor_sample = np.rot90(sensor_sample, k=k, axes=(1, 2))
    return ndvi_sample, sensor_sample


def load_and_reshape_seq(ndvi_folder, sensor_folder, yield_csv_path, target_height=315, target_width=316):
    df_yield = pd.read_csv(yield_csv_path)
    yield_array = df_yield['yield'].values
    num_samples = len(yield_array)

    ndvi_files = sorted([f for f in os.listdir(ndvi_folder) if f.endswith('.npy')])
    sensor_files = sorted([f for f in os.listdir(sensor_folder) if f.endswith('.npy')])

    if len(ndvi_files) % num_samples == 0 and len(sensor_files) % num_samples == 0:
        time_steps_ndvi = len(ndvi_files) // num_samples
        time_steps_sensor = len(sensor_files) // num_samples
        if time_steps_ndvi != time_steps_sensor:
            raise ValueError(f"Mismatch in time steps for NDVI ({time_steps_ndvi}) and sensor ({time_steps_sensor}) files")
        time_steps = time_steps_ndvi
    else:
        raise ValueError("Files count not divisible by number of yield samples.")

    ndvi_data = np.zeros((num_samples, time_steps, target_height, target_width, 1), dtype=np.float32)
    sensor_data = np.zeros((num_samples, time_steps, target_height, target_width, 5), dtype=np.float32)

    for i in range(num_samples):
        for t in range(time_steps):
            ndvi_fp = os.path.join(ndvi_folder, ndvi_files[i * time_steps + t])
            sensor_fp = os.path.join(sensor_folder, sensor_files[i * time_steps + t])

            ndvi_loaded = np.load(ndvi_fp)
            if ndvi_loaded.ndim == 2:
                ndvi_loaded = ndvi_loaded[..., np.newaxis]
            ndvi_loaded = cv2.resize(ndvi_loaded, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            if ndvi_loaded.ndim == 2:
                ndvi_loaded = ndvi_loaded[..., np.newaxis]
            ndvi_data[i, t] = ndvi_loaded.astype(np.float32)

            sensor_loaded = np.load(sensor_fp)
            if sensor_loaded.ndim == 2:
                sensor_loaded = sensor_loaded[..., np.newaxis]
            sensor_loaded = cv2.resize(sensor_loaded, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            if sensor_loaded.ndim == 2:
                sensor_loaded = sensor_loaded[..., np.newaxis]
            sensor_data[i, t] = sensor_loaded.astype(np.float32)

        # ✅ Apply data augmentation here
        ndvi_data[i], sensor_data[i] = augment_sample(ndvi_data[i], sensor_data[i])

    return ndvi_data, sensor_data, yield_array


def preprocess_inputs(ndvi_data, sensor_data):
    ndvi_data = ndvi_data / 255.0
    N, T, H, W, C = sensor_data.shape
    sensor_reshaped = sensor_data.reshape((N * T * H * W, C))
    scaler = StandardScaler()
    sensor_scaled = scaler.fit_transform(sensor_reshaped)
    sensor_scaled = sensor_scaled.reshape((N, T, H, W, C))

    # ✅ Save scaler for later use
    joblib.dump(scaler, 'scaler.save')
    return ndvi_data, sensor_scaled


def build_cnn_lstm_model(ndvi_shape, sensor_shape):
    ndvi_input = Input(shape=ndvi_shape, name='ndvi_input')
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(ndvi_input)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = LSTM(64)(x)

    sensor_input = Input(shape=sensor_shape, name='sensor_input')
    y = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(sensor_input)
    y = TimeDistributed(MaxPooling2D((2, 2)))(y)
    y = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(y)
    y = TimeDistributed(MaxPooling2D((2, 2)))(y)
    y = TimeDistributed(GlobalAveragePooling2D())(y)
    y = LSTM(64)(y)

    combined = Concatenate()([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(1, name='yield_output')(z)

    model = Model(inputs=[ndvi_input, sensor_input], outputs=z)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    return model


def train_and_evaluate(model, ndvi_data, sensor_data, yield_array, test_size=0.2, epochs=50, batch_size=16):
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(ndvi_data, sensor_data, yield_array, test_size=test_size, random_state=42)

    early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        [X1_train, X2_train], y_train,
        validation_data=([X1_val, X2_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early, lr_scheduler]
    )

    loss, mae = model.evaluate([X1_val, X2_val], y_val)
    print(f"Validation loss: {loss:.4f}, MAE: {mae:.4f}")
    return history


def main(ndvi_folder, sensor_folder, yield_csv_path):
    target_height = 315
    target_width = 316

    ndvi_data, sensor_data, yield_array = load_and_reshape_seq(ndvi_folder, sensor_folder, yield_csv_path, target_height, target_width)

    ndvi_data, sensor_data = preprocess_inputs(ndvi_data, sensor_data)

    print(f"NDVI data shape: {ndvi_data.shape}")
    print(f"Sensor data shape: {sensor_data.shape}")
    print(f"Yield shape: {yield_array.shape}")

    ndvi_shape = ndvi_data.shape[1:]
    sensor_shape = sensor_data.shape[1:]
    model = build_cnn_lstm_model(ndvi_shape, sensor_shape)

    train_and_evaluate(model, ndvi_data, sensor_data, yield_array)

    model.save("model.h5")
    print("✅ Model saved as model.h5")


if __name__ == "__main__":
    ndvi_folder = r"data/ndvi"
    sensor_folder = r"data/sensor"
    yield_csv = r"data/yield/Yield_2018To2021.csv"
    main(ndvi_folder, sensor_folder, yield_csv)
