import pandas as pd
import os
import shutil
import glob


results_path = "/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Ultra/results"
algos = ["BiLSTM", "CNNLSTM", "Transformer"]

#Gib den Pfad zur Prediction-Datei an
pred_path = "/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Ultra/results/predictions_all/predictions_BiLSTM__lopo__s1-2-3-4-5-6__pad256_lr001_o5__imuFilt_osimFilt_noYScal_byVar_run09"

# Settings aus Pfad strippen
pred_dirname = os.path.basename(pred_path)
settings = pred_dirname.replace("predictions_", "", 1)
    
# algo-prefix entfernen falls noch vorhanden
algo_prefixes = ["BiLSTM_", "CNNLSTM_", "Transformer_"]
for prefix in algo_prefixes:
    if settings.startswith(prefix):
        settings = settings[len(prefix):]
        break
    
def concat_results(results_path, settings, algos=["BiLSTM", "CNNLSTM", "Transformer"]):
    out_path_metr = os.path.join(results_path, f"metrics_all/metrics_{settings}.csv")
    out_path_pred = os.path.join(results_path, f"predictions_all/predictions_{settings}")

    os.makedirs(os.path.dirname(out_path_metr), exist_ok=True)
    os.makedirs(out_path_pred, exist_ok=True)

    # Führt metricen und plots ordner zusammen
    # Wenn Activitäten/algos getrennt trainiert wurden und results in unterschiedliche Ordner gewandert sind, führt dieses Programm die Ergebnisse in eine Datei zusammen
    #Achtung löscht die zusammengeführten Dateien danach
    all_metrics = []
    first_comment = ""
    for algo in algos:
        pattern_metr = os.path.join(results_path, "metrics_all", f"metrics_{algo}_{settings}.csv")
        pattern_pred = os.path.join(results_path, "predictions_all", f"predictions_{algo}_{settings}", "*.csv")
    
        files_metr = glob.glob(pattern_metr)
        files_pred = glob.glob(pattern_pred)
        for f in files_metr:
            if not first_comment:
                with open(f) as fh:
                    first_line = fh.readline()
                    if first_line.startswith('#'):
                        first_comment = first_line
            df = pd.read_csv(f, comment='#')
            all_metrics.append(df)
            #os.remove(f)  #löschen nach dem Einlesen
        for f in files_pred:
            filename = os.path.basename(f)
        
            dst = os.path.join(out_path_pred, filename)
            shutil.copy2(f, dst)
            print(f"Kopiert: {filename}")
        #Prediction Ordner löschen nachdem alle Dateien kopiert wurden
        pred_folder = os.path.join(results_path, "predictions_all", f"predictions_{algo}_{settings}")
        #if os.path.exists(pred_folder):
        #    shutil.rmtree(pred_folder)
    

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)   
        with open(out_path_metr, 'w') as fh:
            if first_comment:
                fh.write(first_comment)
            combined.to_csv(fh, index=False)
        print(f"Gespeichert: {out_path_metr}")
        print(f"Shape: {combined.shape}")
        print(combined.head())

if __name__ == "__main__":
    concat_results(results_path, settings, algos)