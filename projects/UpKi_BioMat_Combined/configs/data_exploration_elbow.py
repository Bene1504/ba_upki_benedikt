"""
Data exploration plots:
  1. ULTra EF/N – elbow_flex_r + Gyro & Acc of sensor 2 (wrist) and sensor 4 (biceps)
  2. MoSurf     – elbow_flex_r + Gyro & Acc of UA_R (upper arm) and FA_R (forearm)

Run from any directory – adjust the paths at the top if needed.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

SAVE_DIR   = "/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Combined" 

os.makedirs(SAVE_DIR, exist_ok=True)


# HELPER
def time_axis(n, fs=100):
    return np.arange(n) / fs

colors_acc  = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # X, Y, Z
colors_gyro = ["#d62728", "#9467bd", "#8c564b"]



#load ultra EF/N
print("Loading ULTra H5 …")
with open("/home/benedikt_nothhelfer/data_share/ULTra-MoCap/ultramocap_dataset.p", "rb") as f:
    ultra = pickle.load(f)

imu_list_u  = ultra["imu"]
ik_list_u   = ultra["ik"]
meta_list_u = ultra["metadata"]
 
# Find trial: subject_1, movement EF, speed N
trial_idx_u = None
for i, meta in enumerate(meta_list_u):
    subj = meta["subject"].iloc[0]
    mov  = meta["movement"].iloc[0]
    spd  = meta.get("speed", meta.get("trialSpeed", pd.Series(["N"]))).iloc[0]
    if subj == "subject_1" and mov == "EF" and spd == "N":
        trial_idx_u = i
        break
 
if trial_idx_u is None:
    raise ValueError(f"ULTra trial not found: subject_1, EF/N")
 
data_ultra_imu = imu_list_u[trial_idx_u]
data_ultra_ik  = ik_list_u[trial_idx_u]
print(f"ULTra IMU columns: {list(data_ultra_imu.columns)}")
 

sample_cols = list(data_ultra_imu.columns)
print(f"Sample ULTra cols: {sample_cols[:12]}")
 
def ultra_sensor_cols(df, sensor_id, sensor_name_hint):
    features = ["ACCX", "ACCY", "ACCZ", "GYROX", "GYROY", "GYROZ"]
    result = []
    for feat in features:
        candidates = [c for c in df.columns if feat in c and
                      (c.startswith(f"{sensor_id}_") or sensor_name_hint.lower() in c.lower())]
        result.append(candidates[0] if candidates else None)
    return result
 
cols_s2 = ultra_sensor_cols(data_ultra_imu, "2", "wrist")
cols_s4 = ultra_sensor_cols(data_ultra_imu, "4", "biceps")
print(f"Sensor 2 cols: {cols_s2}")
print(f"Sensor 4 cols: {cols_s4}")
 
label_ultra = "elbow_flex_r"
t_ultra = time_axis(len(data_ultra_imu))
 
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f"ULTra – Elbow Flexion / Normal speed  (subject_1)", fontsize=14, fontweight="bold")
 
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
 
ax_label = fig.add_subplot(gs[0, :])
ax_label.plot(t_ultra, data_ultra_ik[label_ultra].values, color="black", linewidth=1.2)
ax_label.set_title("elbow_flex_r  [°]")
ax_label.set_xlabel("Time [s]")
ax_label.set_ylabel("Angle [°]")
ax_label.grid(True, alpha=0.3)
 
sensor_groups = [
    ("Sensor 2 (Wrist)",  cols_s2),
    ("Sensor 4 (Biceps)", cols_s4),
]
 
for col_idx, (sensor_label, feature_cols) in enumerate(sensor_groups):
    acc_cols  = [c for c in feature_cols if c and "ACC"  in c]
    gyro_cols = [c for c in feature_cols if c and "GYRO" in c]
 
    ax_acc = fig.add_subplot(gs[1, col_idx])
    for i, c in enumerate(acc_cols):
        ax_acc.plot(t_ultra, data_ultra_imu[c].values, color=colors_acc[i],
                    linewidth=0.8, label=c[-2:])
    ax_acc.set_title(f"{sensor_label} – Accelerometer  [g]")
    ax_acc.set_xlabel("Time [s]")
    ax_acc.set_ylabel("Acc [g]")
    ax_acc.legend(loc="upper right", fontsize=8)
    ax_acc.grid(True, alpha=0.3)
 
    ax_gyro = fig.add_subplot(gs[2, col_idx])
    for i, c in enumerate(gyro_cols):
        ax_gyro.plot(t_ultra, data_ultra_imu[c].values, color=colors_gyro[i],
                     linewidth=0.8, label=c[-2:])
    ax_gyro.set_title(f"{sensor_label} – Gyroscope  [rad/s]")
    ax_gyro.set_xlabel("Time [s]")
    ax_gyro.set_ylabel("Gyro [rad/s]")
    ax_gyro.legend(loc="upper right", fontsize=8)
    ax_gyro.grid(True, alpha=0.3)
 
out_ultra = os.path.join(SAVE_DIR, "plot_ultra_EF_N_elbow_imu24.png")
fig.savefig(out_ultra, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_ultra}")



# MoSurf
print("Loading MoSurf pickle …")
with open("/home/benedikt_nothhelfer/data_share/MoSurf_Daten/mosurf_dataset.p", "rb") as f:
    surf = pickle.load(f)

imu_list      = surf["imu"]
ik_list       = surf["ik"]
metadata_list = surf["metadata"]
subject_surf  = "AMOAS01"    
activity_surf = "shelve_ordering"  


# Find trial index for the subject + activity
trial_idx = None
for i, meta in enumerate(metadata_list):
    if meta["subject"].iloc[0] == subject_surf and meta["movement"].iloc[0] == activity_surf:
        trial_idx = i
        break

if trial_idx is None:
    raise ValueError(f"Trial not found: subject={subject_surf}, activity={activity_surf}")

imu_df  = imu_list[trial_idx]
ik_df   = ik_list[trial_idx]

print(f"MoSurf IMU columns: {list(imu_df.columns)}")
print(f"MoSurf IK  columns: {list(ik_df.columns)}")

# Detect column naming: could be UA_R_ACCX or RightUpperArm_ACCX etc.
# Try both naming conventions
def find_cols(df, sensor_key, features):
    found = []
    for feat in features:
        candidates = [c for c in df.columns if sensor_key.lower() in c.lower() and feat.lower() in c.lower()]
        if candidates:
            found.append(candidates[0])
        else:
            found.append(None)
    return found


ua_acc  = find_cols(imu_df, "RightUpperArm",  ["ACCX", "ACCY", "ACCZ"])
ua_gyro = find_cols(imu_df, "RightUpperArm",  ["GYROX", "GYROY", "GYROZ"])

fa_acc  = find_cols(imu_df, "RightForearm",  ["ACCX", "ACCY", "ACCZ"])
fa_gyro = find_cols(imu_df, "RightForearm",  ["GYROX", "GYROY", "GYROZ"])

print(f"UA_R acc cols : {ua_acc}")
print(f"UA_R gyro cols: {ua_gyro}")
print(f"FA_R acc cols : {fa_acc}")
print(f"FA_R gyro cols: {fa_gyro}")

# IK label
label_surf = "elbow_flexion_r"
print(f"IK label: {label_surf}")

t_surf = time_axis(len(imu_df))

fig2 = plt.figure(figsize=(16, 10))
title_surf = f"MoSurf – {activity_surf.replace('_',' ').title()}  ({subject_surf})"
fig2.suptitle(title_surf, fontsize=14, fontweight="bold")

gs2 = gridspec.GridSpec(3, 2, figure=fig2, hspace=0.45, wspace=0.35)

# elbow angle 
ax_l2 = fig2.add_subplot(gs2[0, :])
ax_l2.plot(t_surf, ik_df[label_surf].values, color="black", linewidth=1.2)
ax_l2.set_title(f"{label_surf}  [°]")
ax_l2.set_xlabel("Time [s]")
ax_l2.set_ylabel("Angle [°]")
ax_l2.grid(True, alpha=0.3)

sensor_defs = [
    ("Forearm (FA_R)",   fa_acc, fa_gyro),
    ("Upper Arm (UA_R)", ua_acc, ua_gyro),
]

for col_idx, (sensor_label, acc_cols, gyro_cols) in enumerate(sensor_defs):
    # filter out None
    acc_valid  = [c for c in acc_cols  if c is not None]
    gyro_valid = [c for c in gyro_cols if c is not None]

    ax_acc2 = fig2.add_subplot(gs2[1, col_idx])
    for i, c in enumerate(acc_valid):
        ax_acc2.plot(t_surf, imu_df[c].values, color=colors_acc[i],
                     linewidth=0.8, label=c.split("_")[-1])
    ax_acc2.set_title(f"{sensor_label} – Accelerometer  [g]")
    ax_acc2.set_xlabel("Time [s]")
    ax_acc2.set_ylabel("Acc [g]")
    ax_acc2.legend(loc="upper right", fontsize=8)
    ax_acc2.grid(True, alpha=0.3)

    ax_gyro2 = fig2.add_subplot(gs2[2, col_idx])
    for i, c in enumerate(gyro_valid):
        ax_gyro2.plot(t_surf, imu_df[c].values, color=colors_gyro[i],
                      linewidth=0.8, label=c.split("_")[-1])
    ax_gyro2.set_title(f"{sensor_label} – Gyroscope  [rad/s]")
    ax_gyro2.set_xlabel("Time [s]")
    ax_gyro2.set_ylabel("Gyro [rad/s]")
    ax_gyro2.legend(loc="upper right", fontsize=8)
    ax_gyro2.grid(True, alpha=0.3)

out_surf = os.path.join(SAVE_DIR, f"plot_surf_{activity_surf}_elbow.png")
fig2.savefig(out_surf, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out_surf}")
print("Done.")