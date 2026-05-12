#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:43:33 2026

@author: lalesuper
"""
from dataset import DataSet
from config import get_config_universal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

JOINT_DISPLAY_NAMES_ULTRA = {
    "arm_flex_r":   "Shoulder Flexion [°]",
    "arm_add_r":    "Shoulder Adduction [°]",
    "arm_rot_r":    "Shoulder Rotation [°]",
    "elbow_flex_r": "Elbow Flexion [°]",
    "pro_sup_r":    "Forearm Pro/Supination [°]",
    "wrist_flex_r": "Wrist Flexion [°]",
    "wrist_dev_r":  "Wrist Deviation [°]",
}

plt.close('all')
def inspect_original_IMU(normalize=False, unit='si'):
    # unit: 'raw'  → keine Umrechnung (g und rad/s)
    # unit: 'si'   → m/s² und deg/s


    config = get_config_universal('ultramocap', config_path=None)
    dataset_handler = DataSet(config, load_dataset=True)
    x      = dataset_handler.x
    y      = dataset_handler.y
    labels = dataset_handler.labels

    subject_list     =     subject_list    = ["subject_1","subject_2","subject_3","subject_4","subject_5","subject_6",
                           "subject_7","subject_8","subject_9","subject_10","subject_11","subject_12","subject_13"]

    selected_sensors = config["selected_sensors"]
    sensor_to_plot = [4]
    sensor_names = config["sensor_name-"][4]
    n_features       = len(config["selected_imu_features"])  # 6

    speed    = 'N'
    movement = 'AS'

    # Eindeutige Farbe pro Subject
    n_subjects       = len(subject_list)
    colors           = cm.tab20(np.linspace(0, 1, n_subjects))
    subject_color_map = {subject: colors[i] for i, subject in enumerate(subject_list)}

    def normalize_signal(signal):
        return (signal - signal.mean()) / (signal.std() + 1e-8)

    for s_idx, sensor_id in enumerate(selected_sensors):
        if s_idx not in sensor_to_plot:
            continue
        base     = s_idx * n_features
        acc_idx  = [base + 0, base + 1, base + 2]
        gyro_idx = [base + 3, base + 4, base + 5]
        sensor_label = sensor_names if s_idx < len(sensor_names) else f"Sensor {sensor_id}"

        # Umrechnungsfaktoren
        G_TO_MS2  = 9.81   # g → m/s²
        RAD_TO_DEG = 57.2958  # rad/s → deg/s
        
        if unit == 'si':
            acc_labels  = ["ACC X [m/s²]", "ACC Y [m/s²]", "ACC Z [m/s²]"]
            gyro_labels = ["GYRO X [deg/s]", "GYRO Y [deg/s]", "GYRO Z [deg/s]"]
        else:
            acc_labels  = ["ACC X [g]", "ACC Y [g]", "ACC Z [g]"]
            gyro_labels = ["GYRO X [rad/s]", "GYRO Y [rad/s]", "GYRO Z [rad/s]"]

        norm_str = " [normalized]" if normalize else ""
        # Subjects in Gruppen aufteilen (4, 4, 5)
        def chunk_subjects(subject_list, chunk_size=4):
            return [subject_list[i:i+chunk_size] for i in range(0, len(subject_list), chunk_size)]
        
        subject_groups = chunk_subjects(subject_list, chunk_size=5)
        # → [[s1,s2,s3,s4], [s5,s6,s7,s8], [s9,s10,s11,s12,s13]]
        
        for group_idx, subject_group in enumerate(subject_groups):
            
            fig_acc,  axes_acc  = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=False)
            fig_gyro, axes_gyro = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=False)

            #titles not wanted for BA
            #fig_acc.suptitle(f"ACC{norm_str}  |  {sensor_label}  |  {movement}  |  {speed}  |  Group {group_idx+1}",
            #                 fontsize=13, fontweight="bold")
            #fig_gyro.suptitle(f"GYRO{norm_str}  |  {sensor_label}  |  {movement}  |  {speed}  |  Group {group_idx+1}",
            #                  fontsize=13, fontweight="bold")
        
            for subject in subject_group:  # ← nur Subjects dieser Gruppe
                for i in range(len(labels)):
                    if (labels[i]['subject'][0] == subject and      # ← exakter Subject-Match
                        labels[i]['movement'][0] == movement and
                        labels[i]['speed'][0]    == speed):
    
                
    
                   
                        color   = subject_color_map[subject]
                        x_axis = np.arange(x[i].shape[0]) / 100.0
                        subject_display = subject.replace("subject_", "P")  # "subject_1" → "P1"
    
                        for row, (ch_idx, ch_label) in enumerate(zip(acc_idx, acc_labels)):
                            signal = x[i][:, ch_idx]
                            if unit == 'si':
                                signal = signal * G_TO_MS2   # ← g → m/s²
                            if normalize:
                                signal = normalize_signal(signal)
                            
                            axes_acc[row].plot(x_axis, signal, label=subject_display,
                                               color=color, alpha=0.9, linewidth=1.5)
                            axes_acc[row].set_ylabel(ch_label, fontsize=10, fontweight='bold')
                            axes_acc[row].grid(True, linestyle="--", alpha=0.4)
                            axes_acc[row].spines[["top", "right"]].set_visible(False)
                            axes_acc[row].tick_params(labelsize=10)
                            for tick in axes_acc[row].get_yticklabels():
                                tick.set_fontweight('bold')
        
                        for row, (ch_idx, ch_label) in enumerate(zip(gyro_idx, gyro_labels)):
                            signal = x[i][:, ch_idx]
                            if unit == 'si':
                                signal = signal * RAD_TO_DEG  # ← rad/s → deg/s
                            if normalize:
                                signal = normalize_signal(signal)
                            axes_gyro[row].plot(x_axis, signal, label=subject_display,
                                                color=color, alpha=0.9, linewidth=1.5)
                            axes_gyro[row].set_ylabel(ch_label, fontsize=10, fontweight='bold')
                            axes_gyro[row].grid(True, linestyle="--", alpha=0.4)
                            axes_gyro[row].spines[["top", "right"]].set_visible(False)
                            axes_gyro[row].tick_params(labelsize=10)
                            for tick in axes_gyro[row].get_yticklabels():
                                tick.set_fontweight('bold')
    
            # Handles holen für gemeinsame Legende
            handles_acc,  labels_leg_acc  = axes_acc[1].get_legend_handles_labels()
            handles_gyro, labels_leg_gyro = axes_gyro[1].get_legend_handles_labels()
    
            # Duplikate entfernen (falls Subject mehrfach geplottet)
            seen = {}
            unique_handles_acc, unique_labels_acc = [], []
            for h, l in zip(handles_acc, labels_leg_acc):
                if l not in seen:
                    seen[l] = True
                    unique_handles_acc.append(h)
                    unique_labels_acc.append(l)
    
            seen = {}
            unique_handles_gyro, unique_labels_gyro = [], []
            for h, l in zip(handles_gyro, labels_leg_gyro):
                if l not in seen:
                    seen[l] = True
                    unique_handles_gyro.append(h)
                    unique_labels_gyro.append(l)
            
            # Legende rechts außerhalb, vertikal zentriert
            leg_acc = fig_acc.legend(unique_handles_acc, unique_labels_acc,
                                      loc="center left", bbox_to_anchor=(0.85, 0.5),
                                      fontsize=10, framealpha=0.8)
            leg_gyro = fig_gyro.legend(unique_handles_gyro, unique_labels_gyro,
                                        loc="center left", bbox_to_anchor=(0.85, 0.5),
                                        fontsize=10, framealpha=0.8)
            for text in leg_acc.get_texts():
                text.set_fontweight('bold')
            for text in leg_gyro.get_texts():
                text.set_fontweight('bold')
    
            axes_acc[-1].set_xlabel("Time [s]", fontsize=12, fontweight='bold')
            axes_gyro[-1].set_xlabel("Time [s]", fontsize=12, fontweight='bold')
            for tick in axes_acc[-1].get_xticklabels():
                tick.set_fontweight('bold')
            for tick in axes_gyro[-1].get_xticklabels():
                tick.set_fontweight('bold')
    
            fig_acc.tight_layout()
            fig_gyro.tight_layout()
            fig_acc.subplots_adjust(right=0.85)
            fig_gyro.subplots_adjust(right=0.85)
            save_dir = '/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Ultra/results/compareplots/IMU'
            os.makedirs(save_dir, exist_ok=True)
            fname_acc  = f"ACC_{sensor_label}_{movement}_{speed}_group{group_idx+1}.svg"
            fname_gyro = f"GYRO_{sensor_label}_{movement}_{speed}_group{group_idx+1}.svg"
            fig_acc.savefig(os.path.join(save_dir, fname_acc), bbox_inches='tight')
            fig_gyro.savefig(os.path.join(save_dir, fname_gyro), bbox_inches='tight')
            plt.close(fig_acc)
            plt.close(fig_gyro)
            print(f"Saved: {fname_acc}, {fname_gyro}")
            
def inspect_original_Angles(normalize=False):

    config = get_config_universal('ultramocap', config_path=None)
    dataset_handler = DataSet(config, load_dataset=True)

    y      = dataset_handler.y
    labels = dataset_handler.labels

    subject_list    = ["subject_1","subject_2","subject_3","subject_4","subject_5","subject_6",
                       "subject_7","subject_8","subject_9","subject_10","subject_11","subject_12","subject_13"]
    selected_angles = config["selected_opensim_labels"]  # Liste der 7 Winkelnamen

    speed    = 'N'
    movement = 'AS'

    n_subjects        = len(subject_list)
    colors            = cm.tab20(np.linspace(0, 1, n_subjects))
    subject_color_map = {subject: colors[i] for i, subject in enumerate(subject_list)}

    def normalize_signal(signal):
        return (signal - signal.mean()) / (signal.std() + 1e-8)

    def chunk_subjects(subject_list, chunk_size=5):
        return [subject_list[i:i+chunk_size] for i in range(0, len(subject_list), chunk_size)]

    subject_groups = chunk_subjects(subject_list, chunk_size=5)

    # ── Pro Winkel einen eigenen Plot ──────────────────────────────────────────
    for s_idx, angle_name in enumerate(selected_angles):  # ← Fix: enumerate korrekt
        
        norm_str = " [normalized]" if normalize else ""

        for group_idx, subject_group in enumerate(subject_groups):  # ← Fix: Einrückung

            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            #title not wanted in ba
            #fig.suptitle(f"Angle{norm_str}  |  {angle_name}  |  {movement}  |  {speed}  |  Group {group_idx+1}",fontsize=13, fontweight="bold")

            for subject in subject_group:
                for i in range(len(labels)):
                    if (labels[i]['subject'][0] == subject and
                        labels[i]['movement'][0] == movement and
                        labels[i]['speed'][0]    == speed):

                        color   = subject_color_map[subject]
                        x_axis = np.arange(y[i].shape[0]) / 100.0 
                        subject_display = subject.replace("subject_", "P")

                        signal = y[i][:, s_idx]  # ← y statt x, Winkel-Index
                        if normalize:
                            signal = normalize_signal(signal)

                        ax.plot(x_axis, signal,
                                label=subject_display,
                                color=color, alpha=0.9, linewidth=1.5)

            display_name = JOINT_DISPLAY_NAMES_ULTRA.get(angle_name, f"{angle_name} [°]")
            ax.set_ylabel(display_name, fontsize=12, fontweight='bold')
            ax.set_xlabel("Time [s]", fontsize=12, fontweight='bold')
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=10)
            for tick in ax.get_yticklabels():
                tick.set_fontweight('bold')
            for tick in ax.get_xticklabels():
                tick.set_fontweight('bold')

            # Legende deduplizieren
            handles, leg_labels = ax.get_legend_handles_labels()
            seen, unique_h, unique_l = {}, [], []
            for h, l in zip(handles, leg_labels):
                if l not in seen:
                    seen[l] = True
                    unique_h.append(h)
                    unique_l.append(l)

            fig.legend(unique_h, unique_l,
                       loc="center left", bbox_to_anchor=(0.85, 0.5),
                       fontsize=10, framealpha=0.8)
            for text in fig.legends[-1].get_texts():
                text.set_fontweight('bold')

            fig.tight_layout()
            fig.subplots_adjust(right=0.85)
            save_dir = '/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Ultra/results/compareplots/Angles'
            os.makedirs(save_dir, exist_ok=True)
            fname = f"Angle_{angle_name}_{movement}_{speed}_group{group_idx+1}.svg"
            fig.savefig(os.path.join(save_dir, fname), bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {fname}")
            

        
inspect_original_IMU()
inspect_original_Angles()