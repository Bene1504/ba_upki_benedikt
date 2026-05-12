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
from scipy.stats import pearsonr
from concat_results import concat_results

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
    "arm_flex_r":    "Arm Flexion [°]",
    "arm_add_r":     "Arm Adduction [°]",
    "arm_rot_r":     "Arm Rotation [°]",
    "elbow_flex_r":  "Elbow Flexion [°]",
    "pro_sup_r":     "Forearm Pro/Sup [°]",
    "wrist_flex_r":  "Wrist Flexion [°]",
    "wrist_dev_r":   "Wrist Deviation [°]",
}

SAMPLING_FREQ = 100  
SPEED_DURATION_S   = 30    
SPEED_SAMPLES      = SAMPLING_FREQ * SPEED_DURATION_S  
METRICS = ["RMSE", "MAE", "r"]

def run_main():
    dataset_name = 'ultramocap'
    config = get_config_universal(dataset_name)

    
    #Gib den Pfad zur Prediction-Datei an
    pred_path = "/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Ultra/results/predictions_all/predictions_lopo_train_AS_CB_EF_ER_OR_spd_F_N_S_90_180_M_test_AS_CB_EF_ER_OR_spd_N_90_180_M_s1-2-5-6_pad256_lr001_o5_bs32_imuFilt_osimFilt_yScal_HierarchicalScalerbyVar"

    # Settings aus Pfad strippen
    pred_dirname = os.path.basename(pred_path)
    settings = pred_dirname.replace("predictions_", "", 1)
    
    # algo-prefix entfernen falls noch vorhanden
    algo_prefixes = ["BiLSTM_", "CNNLSTM_", "Transformer_"]
    for prefix in algo_prefixes:
        if settings.startswith(prefix):
            settings = settings[len(prefix):]
            break

    results_dir = os.path.expanduser(config["results_path"])
    baseline_metrics_csv = os.path.join(results_dir,"metrics_all", f"metrics_ultra_22.3_lrate2.csv")
    metrics_csv = os.path.join(results_dir,"metrics_all", f"metrics_{settings}.csv")
    pred_dir    = os.path.join(results_dir,"predictions_all", f"predictions_{settings}")
    plots_dir   = os.path.join(results_dir, "plots_all",f"plots_{settings}")
    excel_dir = os.path.join(results_dir,"tables_all")
    excel_out   = os.path.join(excel_dir, f"results_table_{settings}.xlsx")
    
     # Prüfen ob bereits zusammengeführt, falls nicht concat_results ausführen
    #if not os.path.exists(pred_dir) or not os.path.exists(metrics_csv):
    #    print("[INFO] Starte concat_results...")
    #    concat_results(results_dir, settings)
    #else:
    #    print("[INFO] Bereits zusammengeführt.")

    df = compute_metrics_from_predictions(pred_dir)
    
    
    #set true if comparison with baseline is wanted
    comparison_table = True
    comparison_table_vertical = True

#--temporär für visualisieren von finalen 10runs
    settings = "predictions__lopo__s1-2-3-4-5-6__pad256_lr001_o5__imuFilt_osimFilt_noYScal_byVar"
    df, pred_dir = aggregate_runs(pred_dir_base=os.path.join(results_dir, "predictions_all"), settings= settings,dataset_name=dataset_name)
    
    excel_out   = os.path.join(excel_dir, f"results_table_{settings}.xlsx")
    plots_dir   = os.path.join(results_dir, "plots_all",f"plots_{settings}")
#--temporär

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(excel_dir, exist_ok=True)
     
    if comparison_table == False:
        build_excel_table(metrics_csv, excel_out)
    else:
        if comparison_table_vertical == True:
            build_comparison_excel_vertical(df, df, excel_out, settings,label_a="gleiche df mit absichtumerror zu umgehen" )
        else:
            build_comparison_excel_horizontal(baseline_metrics_csv, metrics_csv, excel_out, settings,label_a="ultra_baseline" )
    
    
    build_line_plots(pred_dir, plots_dir, settings, mode="all")

    #fileinfo2= "ultra_22.3_lrate2"
    #metrics_csv2 = os.path.join(results_dir,"metrics_all", f"metrics_{fileinfo2}.csv")
    #plots_dir2   = os.path.join(results_dir, "plots_all/lrate",f"plots_{fileinfo2}")
    #pred_dir2    = os.path.join(results_dir,"predictions_all", f"predictions_{fileinfo2}")
    #os.makedirs(plots_dir2, exist_ok=True)
    #build_line_plots(pred_dir2, plots_dir2, fileinfo2, mode="fast")
    
    
    

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
    if len(sub) == 0:
        return None
    val = sub[metric].iloc[0]
    if isinstance(val, str):
        if len(sub) == 1:
            return val
        # mehrere "mean ± std" Strings → Mittelwert der Mittelwerte + Std der Mittelwerte
        means = []
        for v in sub[metric]:
            try:
                means.append(float(v.split("±")[0].strip()))
            except:
                pass
        if means:
            mean_of_means = round(np.mean(means), 2)
            std_of_means  = round(np.std(means), 2)
            return f"{mean_of_means:.2f} ± {std_of_means:.2f}"
        return val
    return round(float(sub[metric].mean()), 2)

def _write_config_row(ws, config_info, start_row=4, start_col=1):
    if config_info is None:
        return
    
    if isinstance(config_info, str) and os.path.exists(config_info):
        with open(config_info) as f:
            first_line = f.readline()
            if first_line.startswith('#'):
                config_text = first_line.strip().lstrip('#').strip()
            else:
                config_text = ""
    else:
        config_text = config_info  # direkt als Text verwenden

    if config_text:
        cell = ws.cell(row=start_row, column=start_col, value=config_text)
        cell.font = Font(italic=True, size=8, color="555555")
        cell.fill = PatternFill("solid", fgColor="ECEFF1")
        cell.border = _thin_border()
        cell.alignment = Alignment(horizontal="center", vertical="center")
        end_col = ws.max_column
        if end_col > start_col:
            ws.merge_cells(start_row=start_row, start_column=start_col, 
                          end_row=start_row, end_column=end_col)
            
def _write_section_label(ws, label, row, start_col, end_col):
    cell = ws.cell(row=row, column=start_col, value=label)
    cell.font = Font(bold=True, size=11)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.fill = PatternFill("solid", fgColor="ECEFF1")
    cell.border = _thin_border()
    if end_col > start_col:
        ws.merge_cells(start_row=row, start_column=start_col,
                       end_row=row, end_column=end_col)
        

def aggregate_runs(pred_dir_base, settings,  dataset_name="ultramocap"):
    """
    Findet alle Run-Ordner, berechnet Metriken pro Run über alle Modelle,
    mittelt und gibt zurück:
    - df_metrics: "mean ± std" Strings pro Metrik (für Excel)
    - pred_mean_dir: Pfad zum Ordner mit gemittelten Predictions (für Lineplots)
    """
    # Alle Run-Ordner finden
    pattern = os.path.join(pred_dir_base, f"{settings}*_run*")
    run_dirs = sorted(glob.glob(pattern))

    if not run_dirs:
        print(f"[ERROR] Keine Run-Ordner gefunden in {pred_dir_base}")
        return None, None

    print(f"[INFO] {len(run_dirs)} Run-Ordner gefunden")

    # Metriken pro Run sammeln
    all_metrics = []
    for run_dir in run_dirs:
        df_run = compute_metrics_from_predictions(run_dir, dataset_name)
        if df_run is not None and len(df_run) > 0:
            all_metrics.append(df_run)

    if not all_metrics:
        print(f"[ERROR] Keine Metriken berechnet")
        return None, None

    # Metriken aggregieren: mean ± std
    df_concat = pd.concat(all_metrics, ignore_index=True)

    rows = []
    for (algo, subject, activity, joint), group in df_concat.groupby(["algo", "subject", "activity", "joint"], sort=False):
        row = {"algo": algo, "subject": subject, "activity": activity, "joint": joint}
        for metric in METRICS:
            vals = pd.to_numeric(group[metric], errors='coerce')
            mean_val = round(float(vals.mean()), 2)
            std_val  = round(float(vals.std()), 2)
            row[metric] = f"{mean_val:.2f} ± {std_val:.2f}"
        rows.append(row)

    df_metrics = pd.DataFrame(rows)

    # Zeitliche Mittelung der Predictions
    all_file_indices = []
    for run_dir in run_dirs:
        file_index = _parse_file_index(run_dir, dataset_name)
        all_file_indices.append(file_index)

    all_keys = set()
    for fi in all_file_indices:
        all_keys.update(fi.keys())

    # Gemittelte Predictions in temporären Ordner schreiben
    pred_mean_dir = os.path.join(pred_dir_base, "predictions_mean")
    os.makedirs(pred_mean_dir, exist_ok=True)

    for key in all_keys:
        algo, subject, activity, speed = key

        dfs_runs = []
        for fi in all_file_indices:
            if key in fi:
                dfs_runs.append(pd.read_csv(fi[key]))

        if not dfs_runs:
            continue

        min_len   = min(len(df) for df in dfs_runs)
        stacked   = np.stack([df.iloc[:min_len].values for df in dfs_runs], axis=0)
        mean_vals = stacked.mean(axis=0)

        df_mean = pd.DataFrame(mean_vals, columns=dfs_runs[0].columns)

        safe_speed = str(speed).replace("°", "deg").replace(" ", "_").replace("/", "_")
        fname = f"{algo}_{subject}_{activity}_{safe_speed}.csv"
        df_mean.to_csv(os.path.join(pred_mean_dir, fname), index=False)

    return df_metrics, pred_mean_dir

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
    #_write_config_row(ws, metrics_csv, start_row=4+row_offset, start_col=2)

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
        ws.column_dimensions[get_column_letter(2+ col_offset + ai)].width = 13
    
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
    #_write_config_row(ws, metrics_csv, start_row=5+row_offset, start_col=2)

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
    for col_idx in range(2, 2 + len(activities) * len(algos)*len(METRICS)):
        ws.column_dimensions[get_column_letter(col_idx)].width = 13

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
    #_write_config_row(ws, metrics_csv, start_row=5+row_offset, start_col=3)

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
        ws.column_dimensions[get_column_letter(col_idx)].width = 13

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

def build_comparison_excel_horizontal(metrics_csv_a, metrics_csv_b, excel_out, label_b, label_a):
    """
    Liest zwei metrics CSVs ein und schreibt sie nebeneinander in eine Excel-Datei. horizontal
    label_a / label_b: Beschriftung z.B. "baseline" und "noise_filter"
    """
    df_a = pd.read_csv(metrics_csv_a, comment='#')
    df_b = pd.read_csv(metrics_csv_b, comment='#')
    df_a["subject"] = df_a["subject"].str.replace("subject_", "Participant ", regex=False)
    df_b["subject"] = df_b["subject"].str.replace("subject_", "Participant ", regex=False)

    algos      = sorted(df_a["algo"].unique())
    subjects   = sorted(df_a["subject"].unique())
    activities = sorted(df_a["activity"].unique())
    joints     = list(df_a["joint"].unique())

    # Berechne col_offset für Sheet1/2/3
    offset_sheet1 = 1+ len(algos) * len(METRICS) + 1          # +1 Trennspalte
    offset_sheet2 = 1+ len(activities) * len(algos) * len(METRICS) + 1
    offset_sheet3 = 2 + len(joints) * len(algos) * len(METRICS) + 1

    wb = Workbook()
    wb.remove(wb.active)

    for sheet_fn, args_a, args_b, offset in [
        (_build_sheet1, [algos, subjects, metrics_csv_a],
                        [algos, subjects, metrics_csv_b], offset_sheet1),
        (_build_sheet2, [algos, subjects, activities, metrics_csv_a],
                        [algos, subjects, activities, metrics_csv_b], offset_sheet2),
        (_build_sheet3, [algos, subjects, activities, joints, metrics_csv_a],
                        [algos, subjects, activities, joints, metrics_csv_b], offset_sheet3),]:
        sheet_fn(wb, df_a, *args_a)              # Tabelle A, col_offset=0
        sheet_fn(wb, df_b, *args_b, col_offset=offset)  # Tabelle B rechts

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
    upload_to_gdrive(excel_out)

def build_comparison_excel_vertical(metrics_csv_a, metrics_csv_b, excel_out, label_b, label_a):
    """
    Liest zwei metrics CSVs ein und schreibt sie untereinander in eine Excel-Datei. vertikal
    label_a / label_b: Beschriftung z.B. "baseline" und "noise_filter"
    """
    
    if isinstance(metrics_csv_a, str):
        # alter Pfad-Aufruf
        df_a = pd.read_csv(metrics_csv_a, comment='#')
        df_a["subject"] = df_b["subject"].str.replace("subject_", "Participant ", regex=False)
    else:
        # direkt DataFrame
        df_a = metrics_csv_a
    
    if isinstance(metrics_csv_b, str):
        # alter Pfad-Aufruf
        df_b = pd.read_csv(metrics_csv_b, comment='#')
        df_b["subject"] = df_b["subject"].str.replace("subject_", "Participant ", regex=False)
    else:
        # direkt DataFrame
        df_b = metrics_csv_b
    

    algos      = sorted(df_a["algo"].unique())
    subjects   = sorted(df_a["subject"].unique())
    activities = sorted(df_a["activity"].unique())
    joints     = list(df_a["joint"].unique())

    wb = Workbook()
    wb.remove(wb.active)

    n_subjects   = len(subjects) + 1       
    n_act_rows   = (len(subjects) + 1) * len(activities)
    gap = 2  # Leerzeilen zwischen den Tabellen

    #temporär
    #dataset_name = 'ultramocap'
    #config = get_config_universal(dataset_name)
    #results_dir = os.path.expanduser(config["results_path"])
    #fileinfo2= "ultra_22.3_lrate2"
    #fileinfo3= "ultra_22.3_wlength_olap_comb3"
    #metrics_csv2 = os.path.join(results_dir,"metrics_all", f"metrics_{fileinfo2}.csv")
    #metrics_csv3 = os.path.join(results_dir,"metrics_all", f"metrics_{fileinfo3}.csv")
    #df_2 = pd.read_csv(metrics_csv2, comment='#')
    #df_3 = pd.read_csv(metrics_csv3, comment='#')
    #df_2["subject"] = df_2["subject"].str.replace("subject_", "Participant ", regex=False)
    #df_3["subject"] = df_3["subject"].str.replace("subject_", "Participant ", regex=False)

    # Sheet 1
    _build_sheet1(wb, df_b, algos, subjects, metrics_csv_b, label_b)
    row_b_sheet1 = 1 + 3 + n_subjects + gap   # config(1) + header(2+3) + daten + gap
    #---------------------------------------------------------temporär
    #_build_sheet1(wb, df_2, algos, subjects, metrics_csv2, fileinfo2, row_offset=row_b_sheet1)
    #row_b_sheet1 = 2*(1 + 3 + n_subjects + gap)   # config(1) + header(2+3) + daten + gap
    #_build_sheet1(wb, df_3, algos, subjects, metrics_csv3, fileinfo3, row_offset=row_b_sheet1)
    #row_b_sheet1 = 3*(1 + 3 + n_subjects + gap)   # config(1) + header(2+3) + daten + gap
    #-------------------------------------------------------------
    _build_sheet1(wb, df_a, algos, subjects, metrics_csv_a, label_a, row_offset=row_b_sheet1)

    # Sheet 2
    _build_sheet2(wb, df_b, algos, subjects, activities, metrics_csv_b, label_b)
    row_b_sheet2 = 1 + 4 + n_subjects + gap   # config(1) + header(2+3+4) + daten + gap
    #------------------------------------------------------------temporär
    #_build_sheet2(wb, df_2, algos, subjects, activities, metrics_csv2, fileinfo2, row_offset=row_b_sheet2)
    #row_b_sheet2 = 2*(1 + 4 + n_subjects + gap)   # config(1) + header(2+3+4) + daten + gap
    #_build_sheet2(wb, df_3, algos, subjects, activities, metrics_csv3, fileinfo3, row_offset=row_b_sheet2)
    #row_b_sheet2 = 3*(1 + 4 + n_subjects + gap)   # config(1) + header(2+3+4) + daten + gap
    #-------------------------------------------------------------------------
    _build_sheet2(wb, df_a, algos, subjects, activities, metrics_csv_a, label_a, row_offset=row_b_sheet2)

    # Sheet 3
    _build_sheet3(wb, df_b, algos, subjects, activities, joints, metrics_csv_b, label_b)
    row_b_sheet3 = 1 + 4 + n_act_rows + gap
    #----------------------------------temporär
    #_build_sheet3(wb, df_2, algos, subjects, activities, joints,metrics_csv2, fileinfo2, row_offset=row_b_sheet3)
    #row_b_sheet3 = 2*(1 + 4 + n_act_rows + gap)
    #_build_sheet3(wb, df_3, algos, subjects, activities, joints,metrics_csv3, fileinfo3, row_offset=row_b_sheet3)
    #row_b_sheet3 = 3*(1 + 4 + n_act_rows + gap)
    #--------------------------------------------
    _build_sheet3(wb, df_a, algos, subjects, activities, joints, metrics_csv_a, label_a, row_offset=row_b_sheet3)


    wb.save(excel_out)
    print(f"[Comparison Excel] Gespeichert → {excel_out}")
    upload_to_gdrive(excel_out)    

# Line Plots
def build_line_plots(pred_dir,plots_dir, fileinfo, mode= "all"):
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

    #ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    display_name = JOINT_LABELS.get(joint, f"{joint} [°]")
    ax.set_xlabel("Time [s]", fontsize=12, fontweight='bold')
    ax.set_ylabel(display_name, fontsize=12, fontweight='bold')
    ax.set_xlim(0, t_sec[-1] if len(t_sec) > 0 else 30)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="#CCCCCC")
    ax.grid(True, which="major", linestyle="--", alpha=0.4, color="#AAAAAA")
    ax.grid(True, which="minor", linestyle=":",  alpha=0.2, color="#CCCCCC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')

    plt.tight_layout()

    safe_joint = joint.replace("/", "_").replace(" ", "_")
    if mode == "all":
        fname = f"{subject}_{activity}_{safe_joint}_{speed}.svg"
    else:
        fname = f"{subject}_{activity}_{safe_joint}.svg"

    fig.savefig(os.path.join(plots_dir, fname), bbox_inches="tight")
    plt.close(fig)

def _parse_file_index(pred_dir, dataset_name="ultramocap"):
    """Liest alle Prediction-CSVs ein und indexiert (algo, subject, activity) → filepath."""
    pred_files = glob.glob(os.path.join(pred_dir, "*.csv"))
    file_index: dict[tuple, str] = {}

    for fpath in pred_files:
        basename = os.path.splitext(os.path.basename(fpath))[0]
        parts    = basename.split("_")

        if dataset_name == "mosurf":
            subj_idx = next(
                (i for i, p in enumerate(parts) if p.upper().startswith("AMOAS")),
                None,
            )
            if subj_idx is None:
                print(f"[WARN] Kann Subject nicht parsen: {basename}")
                continue

            algo            = "_".join(parts[:subj_idx])
            subject         = parts[subj_idx]
            remainder       = parts[subj_idx + 1:] 


            activity = "_".join(remainder)
            n = len(remainder)
            for split in range(1, n // 2 + 1):
                first_half  = remainder[:split]
                second_half = remainder[split:2 * split]
                if first_half == second_half:
                    activity = "_".join(first_half)
                    break

            speed = "trial" 

        else:  # ultramocap
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

def compute_metrics_from_predictions(pred_dir, dataset_name="ultramocap"):
    
    rows = []
    file_index = _parse_file_index(pred_dir, dataset_name)

    for (algo, subject, activity, speed), fpath in file_index.items():
        subject_display = subject.replace("AMOAS", "Participant ").replace("subject_", "Participant ")
        
        df = pd.read_csv(fpath)
        
        true_cols  = [c for c in df.columns if c.startswith("true_")]
        joint_names = [c.replace("true_", "") for c in true_cols]
        
        for joint in joint_names:
            y_true = df[f"true_{joint}"].values
            y_pred = df[f"pred_{joint}"].values
            
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            mae  = float(np.mean(np.abs(y_pred - y_true)))
            r    = float(pearsonr(y_pred, y_true)[0]) if len(y_true) > 1 else 0.0
            
            rows.append({
                "algo":     algo,
                "subject":  subject.replace("subject_", "Participant "),
                "activity": activity,
                "joint":    joint,
                "RMSE":     round(rmse, 2),
                "MAE":      round(mae, 2),
                "r":        round(r, 2),
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    run_main()