import glob
import re
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.signal import butter, filtfilt
from utils.read_write import read_csv
import os
import pickle
from config import get_config_universal


def build_imu(imu_df, sensor_map, T):
    cols = []
    for sensor_key, sensor_id in sensor_map.items(): #sensor_key wird absichtlich nicht genutzt
        for feat in features:
            if sensor_id is not None:
                col = f"{sensor_id}_{feat}"
                cols.append(imu_df[col].values if col in imu_df.columns else np.zeros(T))
            else:
                cols.append(np.zeros(T))
    return np.stack(cols, axis=1)   

def build_labels(ik_df, label_map, T):
    cols = []
    for unified in unified_labels:
        src = label_map[unified]
        cols.append(ik_df[src].values if src is not None and src in ik_df.columns else np.zeros(T))
    return np.stack(cols, axis=1)   


def process_dataset(pickle_path, sensor_map, label_map, subject_prefix, dataset_source):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    trials = []
    for imu_df, ik_df, meta_df in zip(data["imu"], data["ik"], data["metadata"]):
        T = len(imu_df)
        X = build_imu(imu_df, sensor_map, T)
        Y = build_labels(ik_df, label_map, T)

        subject   = meta_df["subject"].iloc[0]
        movement  = meta_df["movement"].iloc[0]
        speed     = meta_df["speed"].iloc[0] if "speed" in meta_df.columns else "N/A"

        trials.append({
            "imu":            X,              # np.ndarray (T, 24)
            "ik":             Y,              # np.ndarray (T, 4)
            "subject":        f"{subject_prefix}_{subject}",
            "movement":       movement,
            "speed":          speed,
            "dataset_source": dataset_source,
        })
    return trials


# ===========================
# INITIALIZATION
# ===========================
config = get_config_universal('combined')
ultra_pickle  = os.path.join(config['ultra_dl_dataset_path'],config['ultra_dl_dataset'])
surf_pickle   = os.path.join(config['surf_dl_dataset_path'],config['surf_dl_dataset'])
out_pickle    = os.path.join(config['dl_dataset_path'],config['dl_dataset'])

features = config['selected_imu_features']

unified_imu_cols = (
    [f"UA_R_{f}" for f in features] +   # cols  0-5
    [f"FA_R_{f}" for f in features] +   # cols  6-11
    [f"UA_L_{f}" for f in features] +   # cols 12-17  (Nullen für ULTra)
    [f"FA_L_{f}" for f in features]     # cols 18-23  (Nullen für ULTra)
)

unified_labels = config['selected_opensim_labels']
ultra_sensor_map = config['ultra_sensor_mapping']
surf_sensor_map = config['surf_sensor_mapping']

ultra_label_map = config['ultra_label_mapping']
surf_label_map = config['surf_label_mapping']


print("Lade ULTra-MoCap")
ultra_trials = process_dataset(ultra_pickle, ultra_sensor_map, ultra_label_map,
                                subject_prefix="ultra", dataset_source="ultramocap")

print("Lade MoSurf")
surf_trials  = process_dataset(surf_pickle,  surf_sensor_map,  surf_label_map,
                                subject_prefix="surf",  dataset_source="mosurf")

combined = ultra_trials + surf_trials

#save
#save
out = {
    "imu": [pd.DataFrame(t["imu"], columns=unified_imu_cols)for t in combined],
    "ik": [pd.DataFrame(t["ik"], columns=unified_labels)for t in combined],
    "metadata": [pd.DataFrame({"subject": [t["subject"]] * t["imu"].shape[0],
                            "movement": [t["movement"]] * t["imu"].shape[0],
                            "speed": [t["speed"]] * t["imu"].shape[0],
                            "dataset_source": [t["dataset_source"]] * t["imu"].shape[0],})
                            for t in combined],
    "imu_columns":  unified_imu_cols,
    "ik_columns":   unified_labels,
    "dataset_info": {
        "n_ultra": len(ultra_trials),
        "n_surf":  len(surf_trials),
        "n_total": len(combined),
    }
}

with open(out_pickle, "wb") as f:
    pickle.dump(out, f)

print(f"\nFertig!")
print(f"  ULTra-Trials : {len(ultra_trials)}")
print(f"  MoSurf-Trials: {len(surf_trials)}")
print(f"  Gesamt       : {len(combined)}")
print(f"  IMU-Shape    : {combined[0]['imu'].shape}  → (T, 24)")
print(f"  IK-Shape     : {combined[0]['ik'].shape}   → (T, 4)")
print(f"  Gespeichert  : {out_pickle}")
