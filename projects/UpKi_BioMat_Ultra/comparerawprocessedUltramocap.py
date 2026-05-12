import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

h5_path = '/home/benedikt_nothhelfer/data_share/ULTra-MoCap/ULTra-MoCap-processed/All_subjects_data.h5'
raw_base = '/home/benedikt_nothhelfer/data_share/ULTra-MoCap/ULTra-MoCap-raw-0/P01'

# Mapping H5 movement/speed → Raw CSV Pfad
trials = [
    ('AS', 'F', 'Arm Swing/IMU/P01_ArmSwing_Fast.csv'),
    ('AS', 'N', 'Arm Swing/IMU/P01_ArmSwing_Normal.csv'),
    ('AS', 'S', 'Arm Swing/IMU/P01_ArmSwing_Slow.csv'),
    ('EF', 'F', 'Elbow Flexion/IMU/P01_ElbowFlexion_Fast.csv'),
    ('EF', 'N', 'Elbow Flexion/IMU/P01_ElbowFlexion_Normal.csv'),
]

with h5py.File(h5_path, 'r') as f:
    cols = [c.decode() if isinstance(c, bytes) else c for c in f['subject_1/AS/F'].attrs['column_names']]
    col_idx = cols.index('ACCX1')
    vectorized = np.vectorize(lambda x: float(x) if x != b'' else np.nan)

    fig, axes = plt.subplots(len(trials), 1, figsize=(12, 3 * len(trials)))

    for i, (mov, spd, raw_file) in enumerate(trials):
        ds = f[f'subject_1/{mov}/{spd}']
        data_h5 = vectorized(ds[:])

        data_raw = pd.read_csv(f'{raw_base}/{raw_file}')
        step = 2000 // 100
        data_raw_ds = data_raw.iloc[::step].reset_index(drop=True)

        n = min(len(data_h5), len(data_raw_ds))
        axes[i].plot(data_h5[:n, col_idx], label='H5', alpha=0.7)
        axes[i].plot(data_raw_ds['ACCX1'].values[:n], label='Raw downsampled', alpha=0.7)
        axes[i].set_title(f'{mov}/{spd}')
        axes[i].legend()

plt.tight_layout()
plt.savefig('comparison_multi.png')
print("Gespeichert als comparison_multi.png")





# Mehrere Sensoren vergleichen
sensor_cols = ['ACCX1', 'ACCY1', 'ACCZ1', 'GYROX1', 'ACCX3', 'ACCX6']

with h5py.File(h5_path, 'r') as f:
    cols = [c.decode() if isinstance(c, bytes) else c for c in f['subject_1/AS/F'].attrs['column_names']]
    vectorized = np.vectorize(lambda x: float(x) if x != b'' else np.nan)
    ds = f['subject_1/AS/F']
    data_h5 = vectorized(ds[:])

data_raw = pd.read_csv(f'{raw_base}/Arm Swing/IMU/P01_ArmSwing_Fast.csv')
step = 2000 // 100
data_raw_ds = data_raw.iloc[::step].reset_index(drop=True)

fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(12, 3 * len(sensor_cols)))

for i, sensor in enumerate(sensor_cols):
    col_idx = cols.index(sensor)
    n = min(len(data_h5), len(data_raw_ds))
    axes[i].plot(data_h5[:n, col_idx], label='H5', alpha=0.7)
    axes[i].plot(data_raw_ds[sensor].values[:n], label='Raw downsampled', alpha=0.7)
    axes[i].set_title(sensor)
    axes[i].legend()

plt.tight_layout()
plt.savefig('comparison_sensors.png')
print("Gespeichert als comparison_sensors.png")