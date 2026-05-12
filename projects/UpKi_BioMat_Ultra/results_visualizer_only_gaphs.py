import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from config import get_config_universal
from openpyxl import Workbook
from openpyxl import Workbook as WB2
import subprocess
import os
import pickle
from config import get_config_universal
from loading.loadh5dataset import LoadH5DataSet
import os
import tkinter as tk
from tkinter import messagebox
from config import get_config_universal

plt.close('all')


ALGO_COLORS = {
    "BiLSTM":      "#E53935",  
    "Transformer": "#1E88E5", 
    "CNNLSTM":     "#43A047",   
}
ALGO_FILL_COLORS = {
    "BiLSTM":      "FFCDD2",
    "Transformer": "BBDEFB",
    "CNNLSTM":     "C8E6C9",
}
GT_COLOR       = "#212121"

ACTIVITY_LABELS = {
    "AS":  "Arm Swing",
    "CB":  "Crossbody Reach",
    "EF":  "Elbow Flexion",
    "ER":  "Shoulder Rotation",
    "OR":  "Overhead Reach",
}

ACTIVITY_SPEED_ORDER = {
    "AS": ["F", "N", "S", "VF"],
    "CB": ["F", "N", "S"],
    "EF": ["F", "N", "S"],
    "ER": ["F", "N", "S"],
    "OR": ["180deg", "90deg", "Maxdeg"],
}

SPEED_DISPLAY_LABELS = {
    "F":      "Fast",
    "N":      "Normal",
    "S":      "Slow",
    "VF":     "Very Fast",
    "180deg": "180° ROM",
    "90deg":  "90° ROM",
    "Maxdeg": "Max ROM",
}

METRIC_FILL_COLORS = {
    "RMSE":  "FFF3E0",  # dezentes orange
    "MAE":   "E8F5E9",  # dezentes grün  
    "r":     "E3F2FD",  # dezentes blau
}

JOINT_LABELS = {
    "arm_flex_r":    "Shoulder Flexion",
    "arm_add_r":     "Shoulder Adduction",
    "arm_rot_r":     "Shoulder Rotation",
    "elbow_flex_r":  "Elbow Flexion",
    "pro_sup_r":     "Forearm Pro/Sup",
    "wrist_flex_r":  "Wrist Flexion",
    "wrist_dev_r":   "Wrist Deviation",
}

SAMPLING_FREQ = 100  
SPEED_DURATION_S   = 30    
SPEED_SAMPLES      = SAMPLING_FREQ * SPEED_DURATION_S  
METRICS = ["RMSE", "MAE", "r"]

def select_prediction_subfolders_gui(base_dir, max_select=3):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Ordner nicht gefunden: {base_dir}")

    subdirs = sorted(
        [
            name for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        ]
    )

    if not subdirs:
        raise RuntimeError(f"Keine Unterordner in {base_dir} gefunden.")

    selected_paths = []

    def confirm_selection():
        indices = listbox.curselection()
        if not indices:
            messagebox.showwarning("Keine Auswahl", "Bitte mindestens einen Ordner auswählen.")
            return
        if len(indices) > max_select:
            messagebox.showwarning(
                "Zu viele Ordner",
                f"Bitte höchstens {max_select} Ordner auswählen."
            )
            return

        selected_names = [subdirs[i] for i in indices]
        selected_paths.extend([os.path.join(base_dir, name) for name in selected_names])
        root.destroy()

    root = tk.Tk()
    root.title("Predictions-Unterordner auswählen")
    root.geometry("900x450")

    label = tk.Label(
        root,
        text=f"Wähle bis zu {max_select} Unterordner aus:\n{base_dir}",
        justify="left",
        anchor="w"
    )
    label.pack(fill="x", padx=10, pady=(10, 5))

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    scrollbar = tk.Scrollbar(frame, orient="vertical")
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        frame,
        selectmode=tk.MULTIPLE,
        yscrollcommand=scrollbar.set,
        width=120,
        height=18
    )
    for folder in subdirs:
        listbox.insert(tk.END, folder)

    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    button_frame = tk.Frame(root)
    button_frame.pack(fill="x", padx=10, pady=(0, 10))

    ok_button = tk.Button(button_frame, text="Übernehmen", command=confirm_selection)
    ok_button.pack(side="right", padx=5)

    cancel_button = tk.Button(button_frame, text="Abbrechen", command=root.destroy)
    cancel_button.pack(side="right", padx=5)

    root.mainloop()
    return selected_paths


def run_main():
    dataset_name = 'ultramocap'
    config = get_config_universal(dataset_name)

    

    #achte darauf, dass die fileinfo mit der fileinfo von metrics und predictions, die du plotten/tabellisieren willst, übereinstimmt
    
    #fileinfo = "train_all_test_N_only_AS""

    results_dir = os.path.expanduser(config["results_path"])
    default_pred_base = os.path.join(results_dir, "predictions_all")
    # GUI-Ordnerauswahl
    #pred_dirs = select_prediction_subfolders_gui(default_pred_base, max_select=3)
    #pred_dirs = [os.path.join(default_pred_base, "predictions_BiLSTM_lopo_train_AS_CB_EF_ER_OR_spd_F_N_S_90_180_M_test_AS_CB_EF_ER_OR_spd_N_90_180_M_s1-2-5-6_pad256_lr001_o5_bs32_imuFilt_osimFilt_yScal_HierarchicalScalerbyVar"), 
                #os.path.join(default_pred_base, "predictions_BiLSTM_lopo_train_AS_CB_EF_ER_OR_spd_F_N_S_90_180_M_test_AS_CB_EF_ER_OR_spd_N_90_180_M_s1-2-5-6_pad256_lr001_o5_bs32_imuFilt_osimFilt_yScal_MinMaxScalerbyVar"),
                #os.path.join(default_pred_base, "predictions_BiLSTM_lopo_train_AS_CB_EF_ER_OR_spd_F_N_S_90_180_M_test_AS_CB_EF_ER_OR_spd_N_90_180_M_s1-2-5-6_pad256_lr001_o5_bs32_imuFilt_osimFilt_yScal_StandardScalerbyVar")]
    
    pred_dirs = [os.path.join(default_pred_base, "predictions_BiLSTM_lopo_train_AS_CB_EF_ER_OR_spd_F_N_S_90_180_M_test_AS_CB_EF_ER_OR_spd_N_90_180_M_s1-2-5-6_pad256_lr001_o5_bs32_imuFilt_osimFilt_yScal_HierarchicalScalerbyVar"), 
                os.path.join(default_pred_base, "predictions_BiLSTM_lopo_train_AS_CB_EF_ER_OR_spd_F_N_S_90_180_M_test_AS_CB_EF_ER_OR_spd_N_90_180_M_s1-2-5-6_pad256_lr001_o5_bs32_imuFilt_osimFilt_yScal_MinMaxScalerbyVar"),
                os.path.join(default_pred_base, "predictions_BiLSTM_lopo_train_AS_CB_EF_ER_OR_spd_F_N_S_90_180_M_test_AS_CB_EF_ER_OR_spd_N_90_180_M_s1-2-5-6_pad256_lr001_o5_bs32_imuFilt_osimFilt_yScal_StandardScalerbyVar")]

    
    plots_dir   = os.path.join(results_dir, "plots_all")

 
    
   
    build_line_plots(pred_dirs, plots_dir, mode="all")
       
import re

def extract_model_label_from_path(pred_dir_model: str) -> str:
    """
    Erzeugt einen lesbaren, aber eindeutigen Namen aus dem Ordnerpfad.
    Beispiel:
    .../predictions_Transformer_train_XYZ_lr001_bs32
    -> Transformer | lr001 | bs32

    Falls nichts erkannt wird, wird einfach der Ordnername zurückgegeben.
    """
    folder_name = os.path.basename(os.path.normpath(pred_dir_model))

    # 'predictions_' am Anfang entfernen
    label = re.sub(r"^predictions_", "", folder_name)

    # Modelltyp grob erkennen
    model_type = None
    for candidate in ["CNNLSTM", "BiLSTM", "Transformer"]:
        if candidate.lower() in label.lower():
            model_type = candidate
            break

    # Optional: ein paar wichtige Tags aus dem Ordnernamen extrahieren
    lr_match = re.search(r"_lr([^_]+)", label)
    bs_match = re.search(r"_bs([^_]+)", label)
    scaler_match = re.search(r"_(standard|minmax|robust|hierarchical|quantile|power)[^_]*", label, re.IGNORECASE)

    extras = []
    '''if lr_match:
        extras.append(f"lr={lr_match.group(1)}")
    if bs_match:
        extras.append(f"bs={bs_match.group(1)}")'''
    if scaler_match:
        extras.append(scaler_match.group(1))

    if model_type:
        if extras:
            return f"{model_type} | " + " | ".join(extras)
        return model_type

    return folder_name


# Line Plots
def build_line_plots(pred_dirs, plots_dir, mode="all"):
    dataset_name = 'ultramocap'
    config = get_config_universal(dataset_name)
    model_list = ["CNNLSTM", 'BiLSTM', "Transformer"]
    
    
    all_model_data = {}  # {display_name: file_index}

    for pred_dir_model in pred_dirs:
        file_index = _parse_file_index(pred_dir_model)
        if not file_index:
            print(f"[WARNING] Keine Predictions gefunden in {pred_dir_model}")
            continue

        display_name = extract_model_label_from_path(pred_dir_model)

        # falls derselbe Name doppelt entsteht, eindeutig machen
        original_name = display_name
        counter = 2
        while display_name in all_model_data:
            display_name = f"{original_name} ({counter})"
            counter += 1

        all_model_data[display_name] = file_index

    if not all_model_data:
        print("[ERROR] Keine Predictions gefunden für alle Modelle")
        return

    # Gemeinsame Combos über alle Modelle
    all_combos = set()
    for file_index in all_model_data.values():
        for (_, s, a, sp) in file_index.keys():
            all_combos.add((s, a, sp))
    combos_set = sorted(all_combos)

    if mode == "fast":
        fast_combos = set()
        for (subject, activity, speed) in combos_set:
            order = ACTIVITY_SPEED_ORDER.get(activity, [])
            available_speeds = sorted(
                [sp for (s, a, sp) in combos_set if s == subject and a == activity],
                key=lambda sp: order.index(sp) if sp in order else 999
            )
            if available_speeds and speed == available_speeds[0]:
                fast_combos.add((subject, activity, speed))
        combos = sorted(fast_combos)
    else:
        #combos = sorted(combos_set)[0:1]
        combos = sorted(combos_set)

    # Pro Combo: alle Modelle zusammen plotten
    for (subject, activity, speed) in combos:
        dfs = {}
    
        for model_key, file_index in all_model_data.items():
            for (algo, s, a, sp), fpath in file_index.items():
                if s == subject and a == activity and sp == speed:
                    dfs[model_key] = pd.read_csv(fpath)
    
        if not dfs:
            continue
    
        sample_df = next(iter(dfs.values()))
        true_cols = [c for c in sample_df.columns if c.startswith("true_")]
        joint_names = [c.replace("true_", "") for c in true_cols]

        if not dfs:
            continue

        sample_df = next(iter(dfs.values()))
        true_cols = [c for c in sample_df.columns if c.startswith("true_")]
        joint_names = [c.replace("true_", "") for c in true_cols]

        for joint in joint_names:
            _plot_joint(plots_dir, subject, activity, speed, joint, dfs, mode)



    #plot upload
    #upload_folder_to_gdrive(plots_dir, fileinfo)
def _get_model_type(algo_name: str) -> str:
    name = algo_name.lower()
    if "bilstm" in name:
        return "BiLSTM"
    elif "transformer" in name:
        return "Transformer"
    elif "cnnlstm" in name:
        return "CNNLSTM"
    return "Unknown"

def _get_algo_color(algo_name: str, all_algo_names: list[str]) -> str:
    model_types = [_get_model_type(name) for name in all_algo_names]
    current_type = _get_model_type(algo_name)

    # Prüfen, ob ein Modelltyp mehrfach vorkommt
    has_duplicates = model_types.count(current_type) > 1

    # Fall 1: Modelltyp ist eindeutig -> Standardfarbe nach Modell
    if not has_duplicates:
        if current_type == "BiLSTM":
            return ALGO_COLORS["BiLSTM"]
        elif current_type == "Transformer":
            return ALGO_COLORS["Transformer"]
        elif current_type == "CNNLSTM":
            return ALGO_COLORS["CNNLSTM"]
        else:
            return "#9E9E9E"

    # Fall 2: derselbe Modelltyp kommt mehrfach vor -> innerhalb dieses Typs verschieden färben
    same_type_names = [name for name in all_algo_names if _get_model_type(name) == current_type]
    idx = same_type_names.index(algo_name)

    if idx == 0:
        return "#E53935"   # rot
    elif idx == 1:
        return "#1E88E5"   # blau
    elif idx == 2:
        return "#43A047"   # grün
    else:
        return "#9E9E9E"

def _get_algo_display_name(algo_name: str) -> str:
    return algo_name

'''def _get_algo_display_name(algo_name: str) -> str:
    """Kürzt Namen auf lesbare Labels für die Legende."""
    for key in ALGO_COLORS:
        if key.lower() in algo_name.lower():
            return key
    return algo_name'''

def _plot_joint(plots_dir, subject, activity, speed, joint, dfs, mode):
    fig, ax = plt.subplots(figsize=(12, 4))

    first_df = next(iter(dfs.values()))
    y_true   = first_df[f"true_{joint}"].values
    t_sec    = np.arange(len(y_true)) / SAMPLING_FREQ
    true_dataset=load_original()
    #print(true_dataset['metadata'][17]['subject'][0])# 81 start sub2, 17 sub10
    #print(true_dataset['metadata'][17]['movement'][0])
    #print(true_dataset['metadata'][17]['speed'][0])
    
    #y_true_new= true_dataset['ik'][1][joint].values
    #t_sec_new= true_dataset['metadata'][1]['time'].values #-20 after S1 

    # Ground Truth
    plot_labels = list(dfs.keys())
    ax.plot(t_sec, y_true,
            color=GT_COLOR, linewidth=2.0,
            linestyle="-", label="Ground Truth", zorder=5)
    #ax.plot(t_sec_new, y_true_new,
        #color='pink', linewidth=2.0, linestyle='dashed',
         #label="Ground Truth_ori", zorder=5)
    # Predictions je Algo
    for algo, df in dfs.items():
        y_pred      = df[f"pred_{joint}"].values
        min_len     = min(len(t_sec), len(y_pred))
        display_name = _get_algo_display_name(algo)
        ax.plot(t_sec[:min_len], y_pred[:min_len],
                color=_get_algo_color(algo, plot_labels),
                linewidth=1.5, linestyle="--",
                label=display_name, alpha=0.9)

    # Titel: "Subject 1 | Arm Swing | Shoulder Flexion | Fast"
    subj_label   = subject.replace("subject_", "Participant ")
    act_label    = ACTIVITY_LABELS.get(activity, activity)
    joint_label  = JOINT_LABELS.get(joint, joint)
    speed_label  = SPEED_DISPLAY_LABELS.get(speed, speed)

    if mode == "all":
        title = f"{subj_label}  |  {act_label}  |  {joint_label}  |  {speed_label}"
    else:
        title = f"{subj_label}  |  {act_label}  |  {joint_label}"

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Joint Angle [°]", fontsize=11)
    ax.set_xlim(0, t_sec[-1] if len(t_sec) > 0 else 30)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="#CCCCCC")
    ax.grid(True, which="major", linestyle="--", alpha=0.4, color="#AAAAAA")
    ax.grid(True, which="minor", linestyle=":",  alpha=0.2, color="#CCCCCC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)

    safe_joint = joint.replace("/", "_").replace(" ", "_")
    if mode == "all":
        fname = f"{subject}_{activity}_{safe_joint}_{speed}.png"
    else:
        fname = f"{subject}_{activity}_{safe_joint}.png"

    fig.savefig(os.path.join(plots_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)




def _parse_file_index(pred_dir) -> dict[tuple, str]:
    """Liest alle Prediction-CSVs ein und indexiert (algo, subject, activity) → filepath."""
    pred_files = glob.glob(os.path.join(pred_dir, "*.csv"))
    file_index: dict[tuple, str] = {}
    for fpath in pred_files:
        basename = os.path.splitext(os.path.basename(fpath))[0]
        parts    = basename.split("_")
        speed    = parts[-1]
        activity = parts[-2]
        subject  = "_".join(parts[-4:-2]) 
        algo     = "_".join(parts[:-4])
        file_index[(algo, subject, activity, speed)] = fpath
    return file_index

def  load_original(       ):
    

    config = get_config_universal('ultramocap')
    dataset_file = config['dl_dataset_path']+ config['dl_dataset']
    with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
    return(dataset)



          

if __name__ == "__main__":
    run_main()