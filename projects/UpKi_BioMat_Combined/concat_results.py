import pandas as pd
import os
import shutil
import glob

results_path = "/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Ultra/results"
activities = ["AS", "CB", "EF", "ER", "OR"]


out_path_metr = os.path.join(results_path, "metrics_12.3_trainedonactivities.csv")
out_path_pred = os.path.join(results_path, "predictions_12.3_trainedonactivities")
os.makedirs(out_path_pred, exist_ok=True)

# Führt metricen und plots ordner zusammen
# Wenn Activitäten getrennt trainiert wurden und results in unterschiedliche Ordner gewandert sind, führt dieses Programm die Ergebnisse in eine Datei zusammen
all_metrics = []
for activity in activities:
    pattern_metr = os.path.join(results_path, activity, "metrics_*.csv")
    pattern_pred = os.path.join(results_path, activity, "predictions_*", "*.csv")
    
    files_metr = glob.glob(pattern_metr)
    files_pred = glob.glob(pattern_pred)
    for f in files_metr:
        df = pd.read_csv(f)
        df['activity'] = activity
        all_metrics.append(df)
    for f in files_pred:
        filename = os.path.basename(f)
        
        dst = os.path.join(out_path_pred, filename)
        shutil.copy2(f, dst)
        print(f"Kopiert: {filename}")

if all_metrics:
    combined = pd.concat(all_metrics, ignore_index=True)   
    combined.to_csv(out_path_metr, index=False)
    print(f"Gespeichert: {out_path_metr}")
    print(f"Shape: {combined.shape}")
    print(combined.head())