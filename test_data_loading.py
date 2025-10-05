import os
import numpy as np
import pandas as pd

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

    print(f"Number of yield samples: {num_samples}")
    print(f"Time steps: {time_steps}")
    print(f"NDVI files: {len(ndvi_files)}")
    print(f"Sensor files: {len(sensor_files)}")
    print(f"Expected NDVI files: {expected_ndvi_files}")
    print(f"Expected Sensor files: {expected_sensor_files}")
    
    # Test loading just the first few files to verify dimensions
    for i in range(min(2, num_samples)):  # Just test first 2 samples
        for t in range(min(2, time_steps)):  # Just test first 2 time steps
            ndvi_fp = os.path.join(ndvi_folder, ndvi_files[i * time_steps + t])
            sensor_fp = os.path.join(sensor_folder, sensor_files[i * time_steps + t])

            print(f"\nLoading file {i * time_steps + t}:")
            print(f"NDVI file: {os.path.basename(ndvi_fp)}")
            print(f"Sensor file: {os.path.basename(sensor_fp)}")

            ndvi_resized = np.load(ndvi_fp)
            print(f"NDVI original shape: {ndvi_resized.shape}")
            if ndvi_resized.ndim == 2:
                ndvi_resized = ndvi_resized[..., np.newaxis]
            print(f"NDVI shape after adding channel: {ndvi_resized.shape}")
            
            # This is where the original error occurred - trying to assign to wrong dimensions
            # With our fix, this should work now
            ndvi_data[i, t] = ndvi_resized.astype(np.float32)
            print(f"Successfully assigned to ndvi_data[{i}, {t}]")

            sensor_loaded = np.load(sensor_fp)
            print(f"Sensor original shape: {sensor_loaded.shape}")
            if sensor_loaded.ndim == 2:
                sensor_loaded = sensor_loaded[..., np.newaxis]
            print(f"Sensor shape after adding channel: {sensor_loaded.shape}")
            
            # This is where the original error occurred - trying to assign to wrong dimensions
            # With our fix, this should work now
            sensor_data[i, t] = sensor_loaded.astype(np.float32)
            print(f"Successfully assigned to sensor_data[{i}, {t}]")

    return ndvi_data, sensor_data, yield_array

def main():
    ndvi_folder = r"data\ndvi"
    sensor_folder = r"data\sensor"
    yield_csv_path = r"data\yield\Punjab&UP_Yield_2018To2021.csv"  # Use existing file
    
    print("Testing data loading with corrected dimensions...")
    try:
        ndvi_data, sensor_data, yield_array = load_and_reshape_seq(
            ndvi_folder, sensor_folder, yield_csv_path,
            target_height=315,  # Corrected dimension
            target_width=316
        )
        print("\nSUCCESS: Data loading completed without dimension errors!")
        print(f"NDVI data shape: {ndvi_data.shape}")
        print(f"Sensor data shape: {sensor_data.shape}")
        print(f"Yield array shape: {yield_array.shape}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()