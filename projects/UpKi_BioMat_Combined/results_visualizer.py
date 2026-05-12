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

import subprocess

GDRIVE_REMOTE_ULTRA    = "gdrive:UpKi Results/upki_ultra"
GDRIVE_REMOTE_SURF     = "gdrive:UpKi Results/upki_surf"
GDRIVE_REMOTE_COMBINED = "gdrive:UpKi Results/upki_combined"

GDRIVE_REMOTE = GDRIVE_REMOTE_COMBINED

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
    "AS": "Arm Swing",
    "CB": "Crossbody Reach",
    "EF": "Elbow Flexion",
    "ER": "Shoulder Rotation",
    "OR": "Overhead Reach",
    "picking_up_pen":  "Picking Up Pen",  
    "shelve_ordering": "Shelve Ordering",  
}

ACTIVITY_SPEED_ORDER = {
    "AS": ["F", "N", "S", "VF"],
    "CB": ["F", "N", "S"],
    "EF": ["F", "N", "S"],
    "ER": ["F", "N", "S"],
    "OR": ["180deg", "90deg", "Maxdeg"],
    "picking_up_pen":  ["N_A"], 
    "shelve_ordering": ["N_A"],
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


JOINT_LABELS = {
    "elbow_flex_r": "Elbow Flexion R",
    "pro_sup_r":    "Forearm Pro/Sup R",
    "elbow_flex_l": "Elbow Flexion L",
    "pro_sup_l":    "Forearm Pro/Sup L",
}

SAMPLING_FREQ = 100  
SPEED_DURATION_S   = 30    
SPEED_SAMPLES      = SAMPLING_FREQ * SPEED_DURATION_S  
METRICS = ["RMSE", "MAE", "r"]

def run_main():
    dataset_name = 'combined'
    config = get_config_universal(dataset_name)

    parser = argparse.ArgumentParser(description="Ergebnisse visualisieren")
    parser.add_argument("--excel", action="store_true", help="Excel-Tabelle erstellen")
    parser.add_argument("--plots", nargs="?", const="fast",
                        choices=["fast", "all"],
                        help="Line-Plots: 'fast' (nur erste Geschw.) oder 'all' (alle)")
    args = parser.parse_args()

    #achte darauf, dass die fileinfo mit der fileinfo von metrics und predictions, die du plotten/tabellisieren willst, übereinstimmt
    fileinfo = "15.3_combined"

    results_dir = os.path.expanduser(config["results_path"])
    metrics_csv = os.path.join(results_dir,"metrics_all", f"metrics_{fileinfo}.csv")
    pred_dir    = os.path.join(results_dir,"predictions_all", f"predictions_{fileinfo}")
    plots_dir   = os.path.join(results_dir, "plots_all",f"plots_{fileinfo}")
    excel_dir = os.path.join(results_dir,"tables_all")
    excel_out   = os.path.join(excel_dir, f"results_table_{fileinfo}.xlsx")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(excel_dir, exist_ok=True)
     
    build_excel_table(metrics_csv, excel_out)
    build_line_plots(pred_dir, fileinfo, plots_dir)

def _get_algo_color(algo_name: str) -> str:
    for key, color in ALGO_COLORS.items():
        if key.lower() in algo_name.lower():
            return color
    return "#9E9E9E"

def _get_algo_display_name(algo_name: str) -> str:
    """Kürzt technische Namen auf lesbare Labels für die Legende."""
    for key in ALGO_COLORS:
        if key.lower() in algo_name.lower():
            return key
    return algo_name

def _get_algo_fill(algo_name: str) -> str:
    for key, fill in ALGO_FILL_COLORS.items():
        if key.lower() in algo_name.lower():
            return fill
    return "E0E0E0"

def _thin_border():
    s = Side(style="thin", color="CCCCCC")
    return Border(left=s, right=s, top=s, bottom=s)

def _style_header_cell(cell, text, fill_hex, bold=True, size=10):
    cell.value     = text
    cell.font      = Font(bold=bold, size=size)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.fill      = PatternFill("solid", fgColor=fill_hex)
    cell.border    = _thin_border()

def _style_data_cell(cell, value, fill_hex, bold=False):
    cell.value     = value
    cell.font      = Font(bold=bold)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.fill      = PatternFill("solid", fgColor=fill_hex)
    cell.border    = _thin_border()

def _row_fill(idx):
    """Alternating light/dark row fill based on index."""
    return "F0F4F8" if idx % 2 == 0 else "FAFAFA"

def _mean_fill():
    return "E8F5E9"  # leichtes grün für MEAN-Zeilen


# Excel Sheet 1: Subject × Algo
def _build_sheet1(wb, df, algos, subjects):
    ws = wb.create_sheet("Overview")

    #Header Zeile
    # Zeile 1: Algo-Namen
    for ai, algo in enumerate(algos):
        start_col = 2 + ai * len(METRICS)
        end_col   = start_col + len(METRICS) - 1
        cell = ws.cell(row=1, column=start_col)
        _style_header_cell(cell, algo, ALGO_FILL_COLORS.get(algo, "E0E0E0"), size=11)
        if start_col != end_col:
            ws.merge_cells(start_row=1, start_column=start_col, end_row=1, end_column=end_col)

    # Zeile 2: RMSE / MAE / r
    for ai, algo in enumerate(algos):
        for mi, metric in enumerate(METRICS):
            col = 2 + ai * len(METRICS) + mi
            cell = ws.cell(row=2, column=col)
            _style_header_cell(cell, metric, ALGO_FILL_COLORS.get(algo, "E0E0E0"), size=9)

    subj_header = ws.cell(row=1, column=1)
    _style_header_cell(subj_header, "Subject", "ECEFF1", size=10)
    ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=1)

    #Datenzeilen
    all_subjects_with_mean = subjects + ["MEAN"]
    for ri, subject in enumerate(all_subjects_with_mean):
        row_idx = 3 + ri
        is_mean = subject == "MEAN"
        fill    = _mean_fill() if is_mean else _row_fill(ri)

        subj_cell = ws.cell(row=row_idx, column=1, value=subject)
        subj_cell.font      = Font(bold=is_mean)
        subj_cell.alignment = Alignment(horizontal="center", vertical="center")
        subj_cell.fill      = PatternFill("solid", fgColor=fill)
        subj_cell.border    = _thin_border()

        for ai, algo in enumerate(algos):
            for mi, metric in enumerate(METRICS):
                col = 2 + ai * len(METRICS) + mi
                if is_mean:
                    sub = df[df["algo"] == algo]
                else:
                    sub = df[(df["algo"] == algo) & (df["subject"] == subject)]
                val = round(float(sub[metric].mean()), 4) if len(sub) else None
                _style_data_cell(ws.cell(row=row_idx, column=col), val, fill, bold=is_mean)

    #Spaltenbreiten
    for ai in range(len(algos) * len(METRICS)):
        ws.column_dimensions[get_column_letter(2 + ai)].width = 11
    
    ws.column_dimensions["A"].width = 14

    ws.freeze_panes = "B3"



# Excel Sheet 2: Subject × (Aktivität × Algo)
def _build_sheet2(wb, df, algos, subjects, activities):
    ws = wb.create_sheet("By Activity")

    # Aktivitäts-Header ( über Algos)
    ws.cell(row=1, column=1, value="Subject").font = Font(bold=True)
    ws.cell(row=1, column=1).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(row=1, column=1).fill = PatternFill("solid", fgColor="ECEFF1")
    ws.cell(row=1, column=1).border = _thin_border()
    ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=1)

    for ai, activity in enumerate(activities):
        start_col = 2 + ai * len(algos) * len(METRICS)
        end_col   = start_col + len(algos) * len(METRICS) - 1
        label     = ACTIVITY_LABELS.get(activity, activity)
        cell      = ws.cell(row=1, column=start_col, value=label)
        _style_header_cell(cell, label, "E3F2FD", size=10)
        if start_col != end_col:
            ws.merge_cells(start_row=1, start_column=start_col,
                           end_row=1,   end_column=end_col)

    # metrik-Header
    for ai, activity in enumerate(activities):
        for li, algo in enumerate(algos):
            for mi, metric in enumerate(METRICS):
                col  = 2 + ai * len(algos) * len(METRICS) + li * len(METRICS) + mi
                cell = ws.cell(row=3, column=col)
                _style_header_cell(cell, metric, ALGO_FILL_COLORS.get(algo, "E0E0E0"), size=8)

    # algo-header
    for ai, activity in enumerate(activities):
        for li, algo in enumerate(algos):
            start_col = 2 + ai * len(algos) * len(METRICS) + li * len(METRICS)
            end_col   = start_col + len(METRICS) - 1
            cell = ws.cell(row=2, column=start_col)
            _style_header_cell(cell, _get_algo_display_name(algo), ALGO_FILL_COLORS.get(algo, "E0E0E0"), size=9)
            if start_col != end_col:
                ws.merge_cells(start_row=2, start_column=start_col, end_row=2, end_column=end_col)
    
    ws.merge_cells(start_row=1, start_column=1, end_row=3, end_column=1)

    # Datenzeilen
    all_subjects_with_mean = subjects + ["MEAN"]
    for ri, subject in enumerate(all_subjects_with_mean):
        row_idx = 4 + ri
        is_mean = subject == "MEAN"
        fill    = _mean_fill() if is_mean else _row_fill(ri)

        subj_cell = ws.cell(row=row_idx, column=1, value=subject)
        subj_cell.font      = Font(bold=is_mean)
        subj_cell.alignment = Alignment(horizontal="center", vertical="center")
        subj_cell.fill      = PatternFill("solid", fgColor=fill)
        subj_cell.border    = _thin_border()

        for ai, activity in enumerate(activities):
            for li, algo in enumerate(algos):
                for mi, metric in enumerate(METRICS):
                    col = 2 + ai * len(algos) * len(METRICS) + li * len(METRICS) + mi
                    if is_mean:
                        sub = df[(df["algo"] == algo) & (df["activity"] == activity)]
                    else:
                        sub = df[(df["algo"] == algo) & (df["subject"] == subject) &
                                (df["activity"] == activity)]
                    val = round(float(sub[metric].mean()), 4) if len(sub) else None
                    _style_data_cell(ws.cell(row=row_idx, column=col), val, fill, bold=is_mean)
    # Spaltenbreiten
    ws.column_dimensions["A"].width = 14
    for col_idx in range(2, 2 + len(activities) * len(algos)):
        ws.column_dimensions[get_column_letter(col_idx)].width = 13

    ws.freeze_panes = "B4"


# Excel Sheet 3: (Subject × Aktivität) × (Gelenk x Algo)


def _build_sheet3(wb, df, algos, subjects, activities, joints):
    ws = wb.create_sheet("By Joint")

    # Gelenk-Header ( über Algos)
    ws.cell(row=1, column=1, value="Subject").font = Font(bold=True)
    ws.cell(row=1, column=1).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(row=1, column=1).fill = PatternFill("solid", fgColor="ECEFF1")
    ws.cell(row=1, column=1).border = _thin_border()

    ws.cell(row=1, column=2, value="Activity").font = Font(bold=True)
    ws.cell(row=1, column=2).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(row=1, column=2).fill = PatternFill("solid", fgColor="ECEFF1")
    ws.cell(row=1, column=2).border = _thin_border()

    # Merge Subject+Activity header over 2 rows
    ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=1)
    ws.merge_cells(start_row=1, start_column=2, end_row=2, end_column=2)

    joint_fill_colors = [
        "E8EAF6", "FFF3E0", "E8F5E9", "FCE4EC",
        "E0F7FA", "F3E5F5", "FFFDE7"
    ]
    for ji, joint in enumerate(joints):
        start_col = 3 + ji * len(algos) * len(METRICS)
        end_col   = start_col + len(algos) * len(METRICS) - 1
        fill      = joint_fill_colors[ji % len(joint_fill_colors)]
        cell      = ws.cell(row=1, column=start_col, value=joint)
        _style_header_cell(cell, joint, fill, size=9)
        if start_col != end_col:
            ws.merge_cells(start_row=1, start_column=start_col,
                           end_row=1,   end_column=end_col)

    # metrik-header
    for ji, joint in enumerate(joints):
        for li, algo in enumerate(algos):
            for mi, metric in enumerate(METRICS):
                col  = 3 + ji * len(algos) * len(METRICS) + li * len(METRICS) + mi
                cell = ws.cell(row=3, column=col)
                _style_header_cell(cell, metric,  ALGO_FILL_COLORS.get(algo, "E0E0E0"), size=7)


    # Algo-Header
    for ji, joint in enumerate(joints):
        fill = joint_fill_colors[ji % len(joint_fill_colors)]
        for li, algo in enumerate(algos):
            start_col = 3 + ji * len(algos) * len(METRICS) + li * len(METRICS)
            end_col   = start_col + len(METRICS) - 1
            cell = ws.cell(row=2, column=start_col)
            _style_header_cell(cell, _get_algo_display_name(algo),  ALGO_FILL_COLORS.get(algo, "E0E0E0"), size=8)
            if start_col != end_col:
                ws.merge_cells(start_row=2, start_column=start_col, end_row=2, end_column=end_col)

    ws.merge_cells(start_row=1, start_column=1, end_row=3, end_column=1)
    ws.merge_cells(start_row=1, start_column=2, end_row=3, end_column=2)

    # Datenzeilen 
    row_idx = 4
    prev_subject = None
    subject_toggle = 0

    all_subjects_ext = subjects + ["MEAN"]

    for subject in all_subjects_ext:
        is_mean_subject = subject == "MEAN"

        if subject != prev_subject:
            prev_subject = subject
            subject_toggle = 1 - subject_toggle

        activities_here = activities

        for act_idx, activity in enumerate(activities_here):
            is_first_act = act_idx == 0
            fill = _mean_fill() if is_mean_subject else (
                "EEF2F7" if subject_toggle == 0 else "FAFAFA"
            )

            # Subject-Zelle (nur bei erster Aktivität)
            subj_cell = ws.cell(row=row_idx, column=1,
                                value=subject if is_first_act else "")
            subj_cell.font      = Font(bold=is_mean_subject)
            subj_cell.alignment = Alignment(horizontal="center", vertical="center")
            subj_cell.fill      = PatternFill("solid", fgColor=fill)
            subj_cell.border    = _thin_border()

            # Aktivity-Zelle
            act_label = ACTIVITY_LABELS.get(activity, activity)
            act_cell  = ws.cell(row=row_idx, column=2, value=act_label)
            act_cell.alignment = Alignment(horizontal="center", vertical="center")
            act_cell.fill      = PatternFill("solid", fgColor=fill)
            act_cell.border    = _thin_border()
            act_cell.font      = Font(bold=is_mean_subject)

            # Werte pro Gelenk × Algo
            for ji, joint in enumerate(joints):
                for li, algo in enumerate(algos):
                    for mi, metric in enumerate(METRICS):
                        col = 3 + ji * len(algos) * len(METRICS) + li * len(METRICS) + mi
                        if is_mean_subject:
                            sub = df[(df["algo"] == algo) & (df["activity"] == activity) &
                                    (df["joint"] == joint)]
                        else:
                            sub = df[(df["algo"] == algo) & (df["subject"] == subject) &
                                    (df["activity"] == activity) & (df["joint"] == joint)]
                        val = round(float(sub[metric].mean()), 4) if len(sub) else None
                        _style_data_cell(ws.cell(row=row_idx, column=col), val, fill, bold=is_mean_subject)

            row_idx += 1

        # Subject-Zellen merge
        if not is_mean_subject and len(activities) > 1:
            start_merge = row_idx - len(activities)
            end_merge   = row_idx - 1
            try:
                ws.merge_cells(start_row=start_merge, start_column=1,
                               end_row=end_merge,   end_column=1)
            except Exception:
                pass

    # Spaltenbreiten
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 18
    for col_idx in range(3, 3 + len(joints) * len(algos) * len(METRICS)):
        ws.column_dimensions[get_column_letter(col_idx)].width = 10

    ws.freeze_panes = "C4"




def build_excel_table(metrics_csv, excel_out):
    if not os.path.exists(metrics_csv):
        print(f"[ERROR] Keine metrics.csv gefunden unter {metrics_csv}")
        return

    df         = pd.read_csv(metrics_csv)
    df["subject"] = df["subject"].str.replace("subject_", "Participant ", regex=False)
    algos      = sorted(df["algo"].unique())
    subjects   = sorted(df["subject"].unique())
    activities = sorted(df["activity"].unique())
    joints     = list(df["joint"].unique())   # Reihenfolge aus CSV behalten

    from openpyxl import Workbook
    wb = Workbook()
    
    wb.remove(wb.active)

    _build_sheet1(wb, df, algos, subjects)
    _build_sheet2(wb, df, algos, subjects, activities)
    _build_sheet3(wb, df, algos, subjects, activities, joints)

    wb.save(excel_out)
    print(f"[Excel] Gespeichert → {excel_out}  "
          f"({len(subjects)} Subjects, {len(activities)} Aktivitäten, {len(joints)} Gelenke)")

    # gdrive upload
    upload_to_gdrive(excel_out)

# Line Plots
def build_line_plots(pred_dir,fileinfo,plots_dir, mode: str = "fast"):
    """
    mode = 'fast' : nur erste Geschwindigkeit (1 Plot pro Subject × Aktivität × Gelenk)
    mode = 'all'  : alle Geschwindigkeiten als separate Plots
    """
    file_index = _parse_file_index(pred_dir)
    if not file_index:
        print(f"[ERROR] Keine Predictions gefunden in {pred_dir}")
        return

    combos_set = sorted(set((s, a, sp) for (_, s, a, sp) in file_index.keys()))
    algos = sorted(set(a for (a, _, _, _) in file_index.keys()))

    if mode == "fast":
        # Für jede (subject, activity) nur erste Geschwindigkeit
        fast_combos = set()
        for (subject, activity, speed) in combos_set:
            order       = ACTIVITY_SPEED_ORDER.get(activity, [])
            first_speed = order[0] if order else None
            # Prüfe ob diese speed die erste ist
            available_speeds = sorted(
                [sp for (s, a, sp) in combos_set if s == subject and a == activity],
                key=lambda sp: order.index(sp) if sp in order else 999
            )
            if available_speeds and speed == available_speeds[0]:
                fast_combos.add((subject, activity, speed))
        combos = sorted(fast_combos)
    else:
        combos = sorted(combos_set)

    total = 0
    for (subject, activity, speed) in combos:
        dfs: dict[str, pd.DataFrame] = {}
        for algo in algos:
            key = (algo, subject, activity, speed)
            if key in file_index:
                dfs[algo] = pd.read_csv(file_index[key])

        if not dfs:
            continue

        sample_df   = next(iter(dfs.values()))
        true_cols   = [c for c in sample_df.columns if c.startswith("true_")]
        joint_names = [c.replace("true_", "") for c in true_cols]

        for joint in joint_names:
            _plot_joint(plots_dir, subject, activity, speed, joint, dfs, mode)
            total += 1

    print(f"[Plots] {total} Plots gespeichert → {plots_dir}  (mode={mode})")

    #plot upload
    upload_folder_to_gdrive(plots_dir, fileinfo)

def _plot_joint(plots_dir, subject, activity, speed, joint, dfs, mode):
    fig, ax = plt.subplots(figsize=(12, 4))

    first_df = next(iter(dfs.values()))
    y_true   = first_df[f"true_{joint}"].values
    t_sec    = np.arange(len(y_true)) / SAMPLING_FREQ

    # Ground Truth
    ax.plot(t_sec, y_true,
            color=GT_COLOR, linewidth=2.0,
            linestyle="-", label="Ground Truth", zorder=5)

    # Predictions je Algo
    for algo, df in dfs.items():
        y_pred      = df[f"pred_{joint}"].values
        min_len     = min(len(t_sec), len(y_pred))
        display_name = _get_algo_display_name(algo)
        ax.plot(t_sec[:min_len], y_pred[:min_len],
                color=_get_algo_color(algo),
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
    file_index= {}
    for fpath in pred_files:
        basename = os.path.splitext(os.path.basename(fpath))[0]
        parts    = basename.split("_")
        algo = parts[0]
        source = parts[1] 
        if source == "ultra":
            subject = "_".join(parts[1:4])  
            rest = parts[4:]
        else:  # surf
            subject = "_".join(parts[1:3]) 
            rest = parts[3:]
        speed = rest[-1]
        activity = "_".join(rest[:-1])
        file_index[(algo, subject, activity, speed)] = fpath
    return file_index

# Google Drive Upload
def upload_to_gdrive(local_file_path):
    """
    Lädt eine Datei in einen Google Drive Ordner hoch
    Überschreibt bestehende Datei mit gleichem Namen
    """
    result = subprocess.run(
        ["rclone", "copy", local_file_path, GDRIVE_REMOTE],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"[GDrive] Hochgeladen: {os.path.basename(local_file_path)}")
    else:
        print(f"[GDrive] Fehler: {result.stderr}")

def upload_folder_to_gdrive(local_folder_path, fileinfo):
    """
    Lädt einen Folder in einen Google Drive Ordner hoch
    Überschreibt bestehenden Folder mit gleichem Namen
    """
    remote_subfolder = f"{GDRIVE_REMOTE}/plots_{fileinfo}"
    result = subprocess.run(
        ["rclone", "copy", local_folder_path, remote_subfolder],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"[GDrive] Plots hochgeladen → {remote_subfolder}")
    else:
        print(f"[GDrive] Fehler: {result.stderr}")


if __name__ == "__main__":
    run_main()