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

GDRIVE_REMOTE_ULTRA    = "gdrive:UpKi Results/upki_ultra"
GDRIVE_REMOTE_SURF     = "gdrive:UpKi Results/upki_surf"
GDRIVE_REMOTE_COMBINED = "gdrive:UpKi Results/upki_combined"

GDRIVE_REMOTE = GDRIVE_REMOTE_ULTRA

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

def run_main():
    dataset_name = 'ultramocap'
    config = get_config_universal(dataset_name)



    #achte darauf, dass die fileinfo mit der fileinfo von metrics und predictions, die du plotten/tabellisieren willst, übereinstimmt
    fileinfo = "Transformer_lopo_train_AS_spd_N_90_180_M_test_AS_spd_N_90_180_M_s1-2-5-6_pad256_lr001_o5_bs32_imuFilt_osimFilt_yScal_HierarchicalScalerbyVar"
    
    results_dir = os.path.expanduser(config["results_path"])
    pred_dir    = os.path.join(results_dir,"predictions_all", f"predictions_{fileinfo}")
    metrics_csv = os.path.join(results_dir,"metrics_all", f"metrics_{fileinfo}.csv")
    
    baseline="CNNLSTM_lopo_train_AS_spd_N_90_180_M_test_AS_spd_N_90_180_M_s1-2-5-6_pad256_lr001_o5_bs32_imuFilt_osimFilt_yScal_HierarchicalScalerbyVar"
    baseline_metrics_csv = os.path.join(results_dir,"metrics_all", f"metrics_{baseline}.csv")
    
    fileinfo3=baseline[8:]
    excel_dir = os.path.join(results_dir,"tables_all")
    excel_out   = os.path.join(excel_dir, f"results_table_{fileinfo3}.xlsx")
    
    
    fileinfo2= "BiLSTM_lopo_train_AS_spd_N_90_180_M_test_AS_spd_N_90_180_M_s1-2-5-6_pad256_lr001_o5_bs32_imuFilt_osimFilt_yScal_HierarchicalScalerbyVar"
    metrics_csv2 = os.path.join(results_dir,"metrics_all", f"metrics_{fileinfo2}.csv")   
    pred_dir2    = os.path.join(results_dir,"predictions_all", f"predictions_{fileinfo2}")
    
    
    
    #set true if comparison with baseline is wanted
    comparison_table = False
    comparison_table_vertical = True

    #os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(excel_dir, exist_ok=True)
     
    if comparison_table == True:
        
        build_excel_table(metrics_csv, excel_out)
    else:
        if comparison_table_vertical == True:
            build_comparison_excel_vertical(baseline_metrics_csv, metrics_csv, metrics_csv2,excel_out, fileinfo2, fileinfo,label_a=baseline )
        else:
            build_comparison_excel_horizontal(baseline_metrics_csv, metrics_csv, metrics_csv2,excel_out, fileinfo2, fileinfo,label_a=baseline)
    


   
    

def _get_algo_color(algo_name: str) -> str:
    for key, color in ALGO_COLORS.items():
        if key.lower() in algo_name.lower():
            return color
    return "#9E9E9E"

def _get_algo_display_name(algo_name: str) -> str:
    """Kürzt Namen auf lesbare Labels für die Legende."""
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

def _add_comparison_labels(ws, label_a, label_b, offset_b, n_cols_a, n_cols_b):
    ws.insert_rows(1)
    # Label A über gesamte linke Tabelle
    cell_a = ws.cell(row=1, column=1, value=label_a)
    cell_a.font = Font(bold=True, size=11)
    cell_a.alignment = Alignment(horizontal="center", vertical="center")
    cell_a.fill = PatternFill("solid", fgColor="ECEFF1")
    cell_a.border = _thin_border()
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n_cols_a)

    # Label B über gesamte rechte Tabelle
    cell_b = ws.cell(row=1, column=offset_b, value=label_b)
    cell_b.font = Font(bold=True, size=11)
    cell_b.alignment = Alignment(horizontal="center", vertical="center")
    cell_b.fill = PatternFill("solid", fgColor="ECEFF1")
    cell_b.border = _thin_border()
    ws.merge_cells(start_row=1, start_column=offset_b, end_row=1, end_column=offset_b + n_cols_b - 1)

def _add_thick_separator(ws, sep_col):
    """Setzt dicke rechte Border auf sep_col für alle belegten Zeilen."""
    for row in ws.iter_rows(min_col=sep_col, max_col=sep_col):
        for cell in row:
            b = cell.border
            cell.border = Border(
                left=b.left, top=b.top, bottom=b.bottom,
                right=Side(style="medium", color="444444")
            )

def _val(sub, metric):
    """Rundet auf 2 Nachkommastellen."""
    return round(float(sub[metric].mean()), 2) if len(sub) else None

def _write_config_row(ws, metrics_csv, start_row=4, start_col=1):
    """Liest die Kommentarzeile aus der CSV und schreibt sie in Zeile 4."""
    config_text = ""
    with open(metrics_csv) as f:
        first_line = f.readline()
        if first_line.startswith('#'):
            config_text = first_line.strip().lstrip('#').strip()
    if config_text:
        cell = ws.cell(row=start_row, column=start_col, value=config_text)
        cell.font = Font(italic=True, size=8, color="555555")
        cell.fill = PatternFill("solid", fgColor="ECEFF1")
        cell.border = _thin_border()
        cell.alignment = Alignment(horizontal="center", vertical="center")
        # Merge über alle Spalten
        end_col = ws.max_column
        if end_col > start_col:
            ws.merge_cells(start_row=start_row, start_column=start_col, end_row=start_row, end_column=end_col)

def _write_section_label(ws, label, row, start_col, end_col):
    cell = ws.cell(row=row, column=start_col, value=label)
    cell.font = Font(bold=True, size=11)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.fill = PatternFill("solid", fgColor="ECEFF1")
    cell.border = _thin_border()
    if end_col > start_col:
        ws.merge_cells(start_row=row, start_column=start_col,
                       end_row=row, end_column=end_col)

# Excel Sheet 1: Subject × Algo
def _build_sheet1(wb, df, algos, subjects, metrics_csv,label, col_offset=0, row_offset=0):
    ws = wb.create_sheet("Overview") if (col_offset == 0 and row_offset == 0) else wb["Overview"]
    
    # Zeile 1: Algo-Namen
    for ai, algo in enumerate(algos):
        start_col = 2+ col_offset + ai * len(METRICS)
        start_row = 1+ row_offset
        end_col   = start_col + len(METRICS) - 1
        cell = ws.cell(row=start_row, column=start_col)
        _style_header_cell(cell, _get_algo_display_name(algo), ALGO_FILL_COLORS.get(_get_algo_display_name(algo), "E0E0E0"), size=11)
        if start_col != end_col:
            ws.merge_cells(start_row=start_row, start_column=start_col, end_row=1+row_offset, end_column=end_col)

    # Zeile 2: RMSE / MAE / r
    for ai, algo in enumerate(algos):
        for mi, metric in enumerate(METRICS):
            col = 2 + col_offset+ ai * len(METRICS) + mi
            row = 2 + row_offset
            cell = ws.cell(row=row, column=col)
            _style_header_cell(cell, metric, ALGO_FILL_COLORS.get(_get_algo_display_name(algo), "E0E0E0"), size=9)

    _write_section_label(ws, label, 3+row_offset, 2, 1 + len(algos) * len(METRICS))
    #Header Zeile
    _write_config_row(ws, metrics_csv, start_row=4+row_offset, start_col=2)

    #Subjectspalte
    if col_offset == 0:
        subj_header = ws.cell(row=1+row_offset, column=1+ col_offset)
        _style_header_cell(subj_header, "Subject", "ECEFF1", size=10)
        ws.merge_cells(start_row=1+row_offset, start_column=1 + col_offset, end_row=2+row_offset, end_column=1 + col_offset)

    # Graue Füllung für Spalte A in Header-Zeilen
    for r in range(3 + row_offset, 5 + row_offset):
        cell = ws.cell(row=r, column=1)
        cell.fill = PatternFill("solid", fgColor="ECEFF1")

    #Datenzeilen
    all_subjects_with_mean = subjects + ["MEAN"]
    for ri, subject in enumerate(all_subjects_with_mean):
        row_idx = 5 + ri + row_offset
        is_mean = subject == "MEAN"
        fill    = _mean_fill() if is_mean else _row_fill(ri)

        if col_offset == 0:
            subj_cell = ws.cell(row=row_idx, column=1+ col_offset, value=subject)
            subj_cell.font      = Font(bold=is_mean)
            subj_cell.alignment = Alignment(horizontal="center", vertical="center")
            subj_cell.fill      = PatternFill("solid", fgColor=fill)
            subj_cell.border    = _thin_border()

        for ai, algo in enumerate(algos):
            for mi, metric in enumerate(METRICS):
                col = 2 + col_offset+ ai * len(METRICS) + mi
                if is_mean:
                    sub = df[df["algo"] == algo]
                else:
                    sub = df[(df["algo"] == algo) & (df["subject"] == subject)]
                val = _val(sub, metric)
                _style_data_cell(ws.cell(row=row_idx, column=col), val, fill, bold=is_mean)

    #Spaltenbreiten
    for ai in range(len(algos) * len(METRICS)):
        ws.column_dimensions[get_column_letter(2+ col_offset + ai)].width = 11
    
    ws.column_dimensions["A"].width = 14

    ws.freeze_panes = get_column_letter(2 + col_offset) + "3"

# Excel Sheet 2: Subject × (Aktivität × Algo)
def _build_sheet2(wb, df, algos, subjects, activities, metrics_csv, label, col_offset=0, row_offset=0):
    ws = wb.create_sheet("By Activity") if (col_offset == 0 and row_offset == 0) else wb["By Activity"]
    
    #Subjectspalte
    if col_offset == 0:
        subj_header = ws.cell(row=1+row_offset, column=1+ col_offset)
        _style_header_cell(subj_header, "Subject", "ECEFF1", size=10)
        ws.merge_cells(start_row=1+row_offset, start_column=1 + col_offset, end_row=3+row_offset, end_column=1 + col_offset)

    # aktivität header
    for ai, activity in enumerate(activities):
        start_col = 2 + col_offset+ ai * len(algos) * len(METRICS)
        start_row = 1 + row_offset
        end_row = 1 + row_offset
        end_col   = start_col + len(algos) * len(METRICS) - 1
        act_label = ACTIVITY_LABELS.get(activity, activity)
        cell      = ws.cell(row=1+row_offset, column=start_col, value=label)
        _style_header_cell(cell, act_label, "E3F2FD", size=10)
        if start_col != end_col:
            ws.merge_cells(start_row=start_row, start_column=start_col,end_row=end_row,   end_column=end_col)

    # metrik-Header
    for ai, activity in enumerate(activities):
        for li, algo in enumerate(algos):
            for mi, metric in enumerate(METRICS):
                col  = 2 + col_offset+ ai * len(algos) * len(METRICS) + li * len(METRICS) + mi
                cell = ws.cell(row=3+row_offset, column=col)
                _style_header_cell(cell, metric, ALGO_FILL_COLORS.get(_get_algo_display_name(algo), "E0E0E0"), size=8)

    # algo-header
    for ai, activity in enumerate(activities):
        for li, algo in enumerate(algos):
            start_col = 2 + col_offset+ ai * len(algos) * len(METRICS) + li * len(METRICS)
            end_col   = start_col + len(METRICS) - 1
            cell = ws.cell(row=2+row_offset, column=start_col)
            _style_header_cell(cell, _get_algo_display_name(algo), ALGO_FILL_COLORS.get(_get_algo_display_name(algo), "E0E0E0"), size=9)
            if start_col != end_col:
                ws.merge_cells(start_row=2+row_offset, start_column=start_col, end_row=2+row_offset, end_column=end_col)
    

    _write_section_label(ws, label, 4+row_offset, 2, 1 + len(algos) * len(METRICS)*len(activities))
    #Header Zeile
    _write_config_row(ws, metrics_csv, start_row=5+row_offset, start_col=2)

    # Graue Füllung für Spalte A in Header-Zeilen
    for r in range(4 + row_offset, 6 + row_offset):
        cell = ws.cell(row=r, column=1)
        cell.fill = PatternFill("solid", fgColor="ECEFF1")

    # Datenzeilen
    all_subjects_with_mean = subjects + ["MEAN"]
    for ri, subject in enumerate(all_subjects_with_mean):
        row_idx = 6 + ri+row_offset
        is_mean = subject == "MEAN"
        fill    = _mean_fill() if is_mean else _row_fill(ri)

        if col_offset == 0:
            subj_cell = ws.cell(row=row_idx, column=1+ col_offset, value=subject)
            subj_cell.font      = Font(bold=is_mean)
            subj_cell.alignment = Alignment(horizontal="center", vertical="center")
            subj_cell.fill      = PatternFill("solid", fgColor=fill)
            subj_cell.border    = _thin_border()

        for ai, activity in enumerate(activities):
            for li, algo in enumerate(algos):
                for mi, metric in enumerate(METRICS):
                    col = 2+ col_offset + ai * len(algos) * len(METRICS) + li * len(METRICS) + mi
                    if is_mean:
                        sub = df[(df["algo"] == algo) & (df["activity"] == activity)]
                    else:
                        sub = df[(df["algo"] == algo) & (df["subject"] == subject) &
                                (df["activity"] == activity)]
                    val = _val(sub, metric)
                    _style_data_cell(ws.cell(row=row_idx, column=col), val, fill, bold=is_mean)
    # Spaltenbreiten
    ws.column_dimensions["A"].width = 14
    for col_idx in range(2, 2 + len(activities) * len(algos)):
        ws.column_dimensions[get_column_letter(col_idx)].width = 10

    ws.freeze_panes = get_column_letter(2 + col_offset) +"4"

# Excel Sheet 3: (Subject × Aktivität) × (Gelenk x Algo)
def _build_sheet3(wb, df, algos, subjects, activities, joints, metrics_csv, label, col_offset=0, row_offset=0):
    ws = wb.create_sheet("By Joint") if (col_offset == 0 and row_offset == 0) else wb["By Joint"]

    if col_offset == 0:
        #Subjectspalte
        subj_header = ws.cell(row=1+row_offset, column=1+ col_offset)
        _style_header_cell(subj_header, "Subject", "ECEFF1", size=10)
        ws.merge_cells(start_row=1+row_offset, start_column=1 + col_offset, end_row=3+row_offset, end_column=1 + col_offset)

        activity_header = ws.cell(row=1+row_offset, column=2+ col_offset)
        _style_header_cell(activity_header, "Activity", "ECEFF1", size=10)
        ws.merge_cells(start_row=1+row_offset, start_column=2 + col_offset, end_row=3+row_offset, end_column=2 + col_offset)

    joint_fill_colors = ["E8EAF6", "FFF3E0", "E8F5E9", "FCE4EC","E0F7FA", "F3E5F5", "FFFDE7"]

    #joint-header
    for ji, joint in enumerate(joints):
        start_col = 3+col_offset + ji * len(algos) * len(METRICS)
        end_col   = start_col + len(algos) * len(METRICS) - 1
        fill      = joint_fill_colors[ji % len(joint_fill_colors)]
        cell      = ws.cell(row=1+row_offset, column=start_col, value=joint)
        joint_label = JOINT_LABELS.get(joint, joint)
        _style_header_cell(cell, joint_label, fill, size=9)
        if start_col != end_col:
            ws.merge_cells(start_row=1+row_offset, start_column=start_col,end_row=1+row_offset, end_column=end_col)

    # metrik-header
    for ji, joint in enumerate(joints):
        for li, algo in enumerate(algos):
            for mi, metric in enumerate(METRICS):
                col  = 3+col_offset+ ji * len(algos) * len(METRICS) + li * len(METRICS) + mi
                cell = ws.cell(row=3+row_offset, column=col)
                _style_header_cell(cell, metric,  ALGO_FILL_COLORS.get(_get_algo_display_name(algo), "E0E0E0"), size=7)


    # Algo-Header
    for ji, joint in enumerate(joints):
        fill = joint_fill_colors[ji % len(joint_fill_colors)]
        for li, algo in enumerate(algos):
            start_col = 3+col_offset+ ji * len(algos) * len(METRICS) + li * len(METRICS)
            end_col   = start_col + len(METRICS) - 1
            cell = ws.cell(row=2+row_offset, column=start_col)
            _style_header_cell(cell, _get_algo_display_name(algo),  ALGO_FILL_COLORS.get(_get_algo_display_name(algo), "E0E0E0"), size=8)
            if start_col != end_col:
                ws.merge_cells(start_row=2+row_offset, start_column=start_col, end_row=2+row_offset, end_column=end_col)


    _write_section_label(ws, label, 4+row_offset, 3, 2 + len(joints)* len(algos) * len(METRICS))
    #Header Zeile
    _write_config_row(ws, metrics_csv, start_row=5+row_offset, start_col=3)

    # Graue Füllung für Spalte A in Header-Zeilen
    for r in range(4 + row_offset, 6 + row_offset):
        cell = ws.cell(row=r, column=1)
        cell2 = ws.cell(row=r, column=2)
        cell.fill = PatternFill("solid", fgColor="ECEFF1")
        cell2.fill = PatternFill("solid", fgColor="ECEFF1")

    # Datenzeilen 
    row_idx = 6+row_offset
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
            if col_offset == 0:
                subj_cell = ws.cell(row=row_idx, column=1 + col_offset,value=subject if is_first_act else "")
                subj_cell.font      = Font(bold=is_mean_subject)
                subj_cell.alignment = Alignment(horizontal="center", vertical="center")
                subj_cell.fill      = PatternFill("solid", fgColor=fill)
                subj_cell.border    = _thin_border()

            # Aktivity-Zelle
            act_label = ACTIVITY_LABELS.get(activity, activity)
            act_cell  = ws.cell(row=row_idx, column=2 + col_offset, value=act_label)
            act_cell.alignment = Alignment(horizontal="center", vertical="center")
            act_cell.fill      = PatternFill("solid", fgColor=fill)
            act_cell.border    = _thin_border()
            act_cell.font      = Font(bold=is_mean_subject)

            # Werte pro Gelenk × Algo
            for ji, joint in enumerate(joints):
                for li, algo in enumerate(algos):
                    for mi, metric in enumerate(METRICS):
                        col = 3+col_offset + ji * len(algos) * len(METRICS) + li * len(METRICS) + mi
                        if is_mean_subject:
                            sub = df[(df["algo"] == algo) & (df["activity"] == activity) &
                                    (df["joint"] == joint)]
                        else:
                            sub = df[(df["algo"] == algo) & (df["subject"] == subject) &
                                    (df["activity"] == activity) & (df["joint"] == joint)]
                        val = _val(sub, metric)
                        _style_data_cell(ws.cell(row=row_idx, column=col), val, fill, bold=is_mean_subject)

            row_idx += 1

        # Subject-Zellen merge
        if not is_mean_subject and len(activities) > 1:
            start_merge = row_idx - len(activities)
            end_merge   = row_idx - 1
            try:
                ws.merge_cells(start_row=start_merge, start_column=1+ col_offset,end_row=end_merge,   end_column=1)
            except Exception:
                pass

    # Spaltenbreiten
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 18
    for col_idx in range(3, 3 + len(joints) * len(algos) * len(METRICS)):
        ws.column_dimensions[get_column_letter(col_idx)].width = 10

    ws.freeze_panes = get_column_letter(3 + col_offset) + "4"

def build_excel_table(metrics_csv, excel_out):
    if not os.path.exists(metrics_csv):
        print(f"[ERROR] Keine metrics.csv gefunden unter {metrics_csv}")
        return

    df         = pd.read_csv(metrics_csv, comment='#')
    df["subject"] = df["subject"].str.replace("subject_", "Participant ", regex=False)
    algos      = sorted(df["algo"].unique())
    subjects   = sorted(df["subject"].unique())
    activities = sorted(df["activity"].unique())
    joints     = list(df["joint"].unique())   # Reihenfolge aus CSV behalten

    wb = Workbook()
    
    wb.remove(wb.active)

    _build_sheet1(wb, df, algos, subjects, metrics_csv)
    _build_sheet2(wb, df, algos, subjects, activities, metrics_csv)
    _build_sheet3(wb, df, algos, subjects, activities, joints, metrics_csv)

    wb.save(excel_out)
    print(f"[Excel] Gespeichert → {excel_out}  "
          f"({len(subjects)} Subjects, {len(activities)} Aktivitäten, {len(joints)} Gelenke)")
    
    # gdrive upload
    upload_to_gdrive(excel_out)

def build_comparison_excel_horizontal(metrics_csv_a, metrics_csv_b, metrics_csv_c, excel_out, label_c,label_b, label_a):
    """
    Liest zwei metrics CSVs ein und schreibt sie nebeneinander in eine Excel-Datei. horizontal
    label_a / label_b: Beschriftung z.B. "baseline" und "noise_filter"
    """
    df_a = pd.read_csv(metrics_csv_a, comment='#')
    df_b = pd.read_csv(metrics_csv_b, comment='#')
    df_c = pd.read_csv(metrics_csv_c, comment='#')
    df_a["subject"] = df_a["subject"].str.replace("subject_", "Participant ", regex=False)
    df_b["subject"] = df_b["subject"].str.replace("subject_", "Participant ", regex=False)
    df_c["subject"] = df_c["subject"].str.replace("subject_", "Participant ", regex=False)
    
    algos      = sorted(set(df_a["algo"]))
    algo1      = sorted(set(df_b["algo"]))
    algo2      = sorted(set(df_c["algo"]))
    subjects   = sorted(df_a["subject"].unique())
    activities = sorted(df_a["activity"].unique())
    joints     = list(df_a["joint"].unique())

    # Berechne col_offset für Sheet1/2/3
    offset_sheet1 = 1+ len(algos) * len(METRICS) + 1          # +1 Trennspalte
    offset_sheet2 = 1+ len(activities) * len(algos) * len(METRICS) + 1
    offset_sheet3 = 2 + len(joints) * len(algos) * len(METRICS) + 1

    wb = Workbook()
    wb.remove(wb.active)


    for sheet_fn, args_a, args_b, args_c, offset in [
        (_build_sheet1,
            [algos, subjects, metrics_csv_a, label_a],
            [algo1, subjects, metrics_csv_b, label_b],
            [algo2, subjects, metrics_csv_c, label_c],
            offset_sheet1),
        (_build_sheet2,
            [algos, subjects, activities, metrics_csv_a, label_a],
            [algo1, subjects, activities, metrics_csv_b, label_b],
            [algo2, subjects, activities, metrics_csv_c, label_c],
            offset_sheet2),
        (_build_sheet3,
            [algos, subjects, activities, joints, metrics_csv_a, label_a],
            [algo1, subjects, activities, joints, metrics_csv_b, label_b],
            [algo2, subjects, activities, joints, metrics_csv_c, label_c],
            offset_sheet3),
    ]:
        sheet_fn(wb, df_a, *args_a)
        sheet_fn(wb, df_b, *args_b, col_offset=offset)
        sheet_fn(wb, df_c, *args_c, col_offset=2 * offset)


    # Label-Zeilen und dicke Trennlinie pro Sheet
    for ws_name, offset, n_cols_a, n_cols_b in [("Overview",    offset_sheet1, offset_sheet1 - 1,len(algos) * len(METRICS)),          # ohne Subject-Spalte
                                                ("By Activity", offset_sheet2, offset_sheet2 - 1, len(activities) * len(algos) * len(METRICS)),
                                                ("By Joint",    offset_sheet3,offset_sheet3 - 1,1+len(joints) * len(algos) * len(METRICS)),]:
        ws = wb[ws_name]
        _add_comparison_labels(ws, label_a, label_b, offset, n_cols_a, n_cols_b)
        _add_thick_separator(ws, offset - 1)

    wb.save(excel_out)
    print(f"[Comparison Excel] Gespeichert → {excel_out}")
    # gdrive upload
    #upload_to_gdrive(excel_out)

def build_comparison_excel_vertical(metrics_csv_a, metrics_csv_b, metrics_csv_c, excel_out, label_c,label_b, label_a):
    """
    Liest zwei metrics CSVs ein und schreibt sie untereinander in eine Excel-Datei. vertikal
    label_a / label_b: Beschriftung z.B. "baseline" und "noise_filter"
    """
    df_a = pd.read_csv(metrics_csv_a, comment='#')
    df_b = pd.read_csv(metrics_csv_b, comment='#')
    df_c = pd.read_csv(metrics_csv_c, comment='#')
    df_a["subject"] = df_a["subject"].str.replace("subject_", "Participant ", regex=False)
    df_b["subject"] = df_b["subject"].str.replace("subject_", "Participant ", regex=False)
    df_c["subject"] = df_c["subject"].str.replace("subject_", "Participant ", regex=False)
    
    algos      = sorted(set(df_a["algo"]))
    algo1      = sorted(set(df_b["algo"]))
    algo2      = sorted(set(df_c["algo"]))
    subjects   = sorted(df_a["subject"].unique())
    activities = sorted(df_a["activity"].unique())
    joints     = list(df_a["joint"].unique())

    wb = Workbook()
    wb.remove(wb.active)

    n_subjects   = len(subjects) + 1       
    n_act_rows   = (len(subjects) + 1) * len(activities)
    gap = 2  # Leerzeilen zwischen den Tabellen

    #temporär

    metrics_csv2 = metrics_csv_c

    # Sheet 1
    _build_sheet1(wb, df_b, algo1, subjects, metrics_csv_b, label_b)
    row_b_sheet1 = 1 + 3 + n_subjects + gap   # config(1) + header(2+3) + daten + gap
    #---------------------------------------------------------temporär
    _build_sheet1(wb, df_c, algo2, subjects, metrics_csv_c, label_c, row_offset=row_b_sheet1)
    row_b_sheet1 = 2*(1 + 3 + n_subjects + gap)   # config(1) + header(2+3) + daten + gap
    #-------------------------------------------------------------
    _build_sheet1(wb, df_a, algos, subjects, metrics_csv_a, label_a, row_offset=row_b_sheet1)

    # Sheet 2
    _build_sheet2(wb, df_b, algo1, subjects, activities, metrics_csv_b, label_b)
    row_b_sheet2 = 1 + 4 + n_subjects + gap   # config(1) + header(2+3+4) + daten + gap
    #------------------------------------------------------------temporär
    _build_sheet2(wb, df_c, algo2, subjects, activities, metrics_csv_c, label_c, row_offset=row_b_sheet2)
    row_b_sheet2 = 2*(1 + 4 + n_subjects + gap)   # config(1) + header(2+3+4) + daten + gap
    #-------------------------------------------------------------------------
    _build_sheet2(wb, df_a, algos, subjects, activities, metrics_csv_a, label_a, row_offset=row_b_sheet2)

    # Sheet 3
    _build_sheet3(wb, df_b, algo1, subjects, activities, joints, metrics_csv_b, label_b)
    row_b_sheet3 = 1 + 4 + n_act_rows + gap
    #----------------------------------temporär
    _build_sheet3(wb, df_c, algo2, subjects, activities, joints,metrics_csv2, label_c, row_offset=row_b_sheet3)
    row_b_sheet3 = 2*(1 + 4 + n_act_rows + gap)
    #--------------------------------------------
    _build_sheet3(wb, df_a, algos, subjects, activities, joints, metrics_csv_a, label_a, row_offset=row_b_sheet3)


    wb.save(excel_out)
    print(f"[Comparison Excel] Gespeichert → {excel_out}")
    #upload_to_gdrive(excel_out)    


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