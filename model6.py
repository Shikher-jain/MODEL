import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Concatenate, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2

def load_and_resize_file(filepath, target_shape):
    data = np.load(filepath)
    if data.ndim == 2:
        data_resized = cv2.resize(data, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        return data_resized
    elif data.ndim == 3:
        channels = data.shape[2]
        resized_channels = []
        for c in range(channels):
            resized_c = cv2.resize(data[:, :, c], (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            resized_channels.append(resized_c)
        return np.stack(resized_channels, axis=-1)
    else:
        raise ValueError(f"Unsupported data shape for resizing: {data.shape}")

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

    expected_ndvi_files = num_samples * time_steps
    expected_sensor_files = num_samples * time_steps

    if len(ndvi_files) != expected_ndvi_files or len(sensor_files) != expected_sensor_files:
        raise ValueError("Number of files does not match expected based on time steps and samples.")

    ndvi_data = np.zeros((num_samples, time_steps, target_height, target_width, 1), dtype=np.float32)
    sensor_data = np.zeros((num_samples, time_steps, target_height, target_width, 5), dtype=np.float32)  # 5 channels

    for i in range(num_samples):
        for t in range(time_steps):
            ndvi_fp = os.path.join(ndvi_folder, ndvi_files[i * time_steps + t])
            sensor_fp = os.path.join(sensor_folder, sensor_files[i * time_steps + t])

            ndvi_resized = np.load(ndvi_fp)
            if ndvi_resized.ndim == 2:
                ndvi_resized = ndvi_resized[..., np.newaxis]
            ndvi_data[i, t] = ndvi_resized.astype(np.float32)

            sensor_loaded = np.load(sensor_fp)
            if sensor_loaded.ndim == 2:
                sensor_loaded = sensor_loaded[..., np.newaxis]
            sensor_data[i, t] = sensor_loaded.astype(np.float32)

    return ndvi_data, sensor_data, yield_array

def preprocess_inputs(ndvi_data, sensor_data):
    ndvi_data = ndvi_data / 255.0
    N, T, H, W, C = sensor_data.shape
    sensor_reshaped = sensor_data.reshape((N * T * H * W, C))
    scaler = StandardScaler()
    sensor_scaled = scaler.fit_transform(sensor_reshaped)
    sensor_scaled = sensor_scaled.reshape((N, T, H, W, C))
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
    z = Dropout(0.5)(z)  # Adding dropout layer
    z = Dense(1, name='yield_output')(z)

    model = Model(inputs=[ndvi_input, sensor_input], outputs=z)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    return model

def train_and_evaluate(model, ndvi_data, sensor_data, yield_array, test_size=0.2, epochs=50, batch_size=16):
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        ndvi_data, sensor_data, yield_array, test_size=test_size, random_state=42)

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
    target_height = 315  # Corrected from 316 to 315 to match actual data dimensions
    target_width = 316

    ndvi_data, sensor_data, yield_array = load_and_reshape_seq(ndvi_folder, sensor_folder, yield_csv_path,
                                                               target_height=target_height,
                                                               target_width=target_width)

    ndvi_data, sensor_data = preprocess_inputs(ndvi_data, sensor_data)

    print(f"NDVI data shape: {ndvi_data.shape}")
    print(f"Sensor data shape: {sensor_data.shape}")
    print(f"Yield shape: {yield_array.shape}")

    ndvi_shape = ndvi_data.shape[1:]
    sensor_shape = sensor_data.shape[1:]
    model = build_cnn_lstm_model(ndvi_shape, sensor_shape)

    train_and_evaluate(model, ndvi_data, sensor_data, yield_array)


if __name__ == "__main__":
    ndvi_folder = r"data\ndvi"
    sensor_folder = r"data\sensor"
    yield_csv = r"data\yield\Punjab&UP_Yield_2018To2021.csv"
    main(ndvi_folder, sensor_folder, yield_csv)
