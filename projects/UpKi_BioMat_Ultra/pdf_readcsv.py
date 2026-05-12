import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate

"""Prüft ob die raw daten richtig preprocessed zu den h5daten von ultramocap sind"""

H5_PATH = '/home/benedikt_nothhelfer/data_share/ULTra-MoCap/ULTra-MoCap-processed/All_subjects_data.h5'
RAW_IMU_PATH = '/home/benedikt_nothhelfer/data_share/ULTra-MoCap/ULTra-MoCap-raw-1/P10/Elbow Flexion/IMU/'

def minmax_norm(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

# H5 laden 
with h5py.File(H5_PATH, 'r') as f:
    ds = f['subject_10/EF/N']
    cols = [c.decode() if isinstance(c, bytes) else c for c in ds.attrs['column_names']]
    raw_bytes = ds[:]
    data_h5 = np.where(raw_bytes == b'', np.nan, raw_bytes).astype(float)

df_h5 = pd.DataFrame(data_h5, columns=cols)
print(f"H5 shape: {df_h5.shape}")

#  Raw laden
files = sorted(os.listdir(RAW_IMU_PATH))
print(f"Raw Dateien: {files}")

normal_file = [f for f in files if 'Normal' in f and 'VeryFast' not in f]
print(f"Normal-Kandidat: {normal_file}")

df_raw = pd.read_csv(os.path.join(RAW_IMU_PATH, normal_file[0]))
df_raw_ds = df_raw.iloc[::20].reset_index(drop=True)
print(f"Raw shape: {df_raw.shape} → downsampled: {df_raw_ds.shape}")
print(f"H5 shape:  {df_h5.shape}")

n = min(len(df_h5), len(df_raw_ds))
print(f"Vergleichbare Länge: {n} Samples ({n/100:.1f}s)")

#  Sync-Check
# Raw ab Offset 40000 laden und downsampeln
raw_aligned = df_raw.iloc[40000::20].reset_index(drop=True)
n = min(len(df_h5), len(raw_aligned))

acc_h5  = df_h5['ACCX3'].values[:n]
acc_h5_norm  = (df_h5['ACCX3'].values[:n] - df_h5['ACCX3'].mean()) / df_h5['ACCX3'].std()
acc_raw = df_raw_ds['ACCX3'].values[:n]
acc_raw_norm = (raw_aligned['ACCX3'].values[:n] - raw_aligned['ACCX3'].mean()) / raw_aligned['ACCX3'].std()

ik_signal = df_h5['elbow_flex_r'].values[:n]  # EF → elbow_flex_r passt gut
ik_norm   = (ik_signal - np.nanmean(ik_signal)) / np.nanstd(ik_signal)

def cross_corr(a, b):
    a = np.nan_to_num((a - np.nanmean(a)) / np.nanstd(a))
    b = np.nan_to_num((b - np.nanmean(b)) / np.nanstd(b))
    corr = np.correlate(a, b, mode='full')
    lags = np.arange(-len(a)+1, len(a))
    return lags, corr

lags, corr = cross_corr(acc_h5_norm, ik_norm)
best_lag = lags[np.argmax(corr)]

# Plot
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(acc_h5_norm, label='H5 ACCX3', alpha=0.8)
ax.plot(acc_raw_norm, label='Raw ACCX3 (aligned)', alpha=0.6, linestyle='--')
ax.set_title('H5 vs Raw (korrekt aligned) — subject_10 / EF / N')
ax.legend()
plt.tight_layout()
plt.savefig('/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Ultra/sync_check2.3.png', dpi=150)






