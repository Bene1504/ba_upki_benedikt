import os
import numpy as np
import pandas as pd


class ResultsCollector:
    """
    Sammelt Metriken und Predictions über einen kompletten LOPO-Run
    und speichert sie als CSV. (pro Subject × Aktivität × Algorithmus)
    """
    def __init__(self, config):
        self.algo_name   = config['model_name']
        self._metrics_df : pd.DataFrame | None = None

        #Infos aus der config zum run
        self.learning_rate  = config['learning_rate']
        self.n_epoch        = config['n_epoch']
        self.batch_size     = config['batch_size']
        self.window_size = config['target_padding_length']
        self.overlap = config['overlap']
        self.resample = config['resample']
        self.imu_filter = config['imu_filter']
        self.opensim_filter = config['opensim_filter']
        self.y_scaling = config['y_scaling']

        self.results_dir = os.path.expanduser(config['results_path'])
        #ändere nach Belieben den Namen der predictions/metrics datei
        self.file_info = self.build_file_info(config)
        fileinfo = self.file_info
        self.metr_dir= os.path.join(self.results_dir,"metrics_all")
        self.metrics_csv = os.path.join(self.results_dir,"metrics_all", f"metrics_{fileinfo}.csv")
        self.pred_dir    = os.path.join(self.results_dir,"predictions_all", f"predictions_{fileinfo}")
        
        self.stride = round(self.overlap * self.window_size)
        self.label_idx_subject  = 0
        self.label_idx_activity = 1
        self.label_idx_speed    = 2
        self.label_idx_time     = 3

        os.makedirs(self.metr_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.pred_dir,    exist_ok=True)

    def build_file_info(self, cfg):

        model_by_map={
            "hernandez2021cnnlstm": "CNNLSTM",
            "bilstm": "BiLSTM",
            "transformertsai": "Transformer"}
        model=model_by_map.get(cfg["model_name"])
        style        = cfg["training_style"]
        train_act    = "_".join(cfg["train_activity"])
        test_act     = "_".join(cfg["test_activity"])
        sensors      = "-".join(cfg["selected_sensors"])
        pad          = cfg["target_padding_length"]
        lr           = str(cfg["learning_rate"]).replace("0.", "")
        ovlp         = str(cfg["overlap"]).replace("0.", "")
        #bs           = cfg["batch_size"]
        #loss         = cfg["loss"]
        imu_filt     = "imuFilt"  if cfg.get("imu_filter",     False) else "noImuFilt"
        osim_filt    = "osimFilt" if cfg.get("opensim_filter",  False) else "noOsimFilt"
        y_scal       = "yScal"    if cfg.get("y_scaling",       False) else "noYScal"
        run_idx = cfg.get("run_idx", 0)

        scaler_by_map = {
        "by_groupvar": "grpVar",
        "by_var":      "byVar",
        "by_sample":   "bySamp",
        "by_step":     "byStep",
        }
        scaler_by = scaler_by_map.get(cfg["data_transformer"]["data_transformer_by"], "unkown")
    
        #f"__train_{train_act}"
        #f"__test_{test_act}"
        # _bs{bs}
        return (
            f"{model}__{style}"
            f"__s{sensors}__pad{pad}_lr{lr}_o{ovlp}"
            f"__{imu_filt}_{osim_filt}_{y_scal}_{scaler_by}"
            f"_run{run_idx:02d}"
        )
    
    def log_metrics(self,subject,activity,rmse_all,mae_all,n_rmse_all,r_all,joint_names):
        """
        Fügt Metriken für Subject x Aktivität hinzu.
        """
        new_rows = pd.DataFrame({"algo": self.algo_name,"subject": subject,"activity": activity,"joint": joint_names,"RMSE": [float(v) for v in rmse_all],
            "MAE":      [float(v) for v in mae_all],"nRMSE": [float(v) for v in n_rmse_all],"r": [float(v) for v in r_all],})

        if self._metrics_df is None:
            self._metrics_df = new_rows
        else:
            self._metrics_df = pd.concat(
                [self._metrics_df, new_rows], ignore_index=True
            )

    def log_predictions(self, subject, activity, y_true, y_pred, labels, joint_names):
        """
        Speichert y_true + y_pred für Subject x Aktivitäts.
        """
        n, t, j = y_true.shape

        if joint_names is None:
            joint_names = [f"joint_{k}" for k in range(j)]

        speeds = np.array([str(lbl[0][self.label_idx_speed]) for lbl in labels])
        unique_speeds = list(dict.fromkeys(speeds))  

        for speed in unique_speeds:
            speed_mask  = speeds == speed
            speed_idx   = np.where(speed_mask)[0]

            if len(speed_idx) == 0:
                continue

            y_true_speed = y_true[speed_idx]   
            y_pred_speed = y_pred[speed_idx]

            true_rec = self._overlap_mean_reconstruct(y_true_speed, self.window_size, self.stride)
            pred_rec = self._overlap_mean_reconstruct(y_pred_speed, self.window_size, self.stride)

            df = pd.DataFrame(
                {f"true_{name}": true_rec[:, k] for k, name in enumerate(joint_names)}
                | {f"pred_{name}": pred_rec[:, k] for k, name in enumerate(joint_names)}
            )

            safe_speed = str(speed).replace("°", "deg").replace(" ", "_").replace("/", "_")
            filename   = f"{self.algo_name}_{subject}_{activity}_{safe_speed}.csv"
            df.to_csv(os.path.join(self.pred_dir, filename), index=False)
            

    def save(self):
        """
        Schreibt alle Metriken in metrics.csv datei.
        """
        if self._metrics_df is None:
            print("[ResultsCollector] Keine Metriken zum Speichern.")
            return

        current_combos = set(
            zip(self._metrics_df["algo"], self._metrics_df["subject"])
        )

        if os.path.exists(self.metrics_csv):
            existing = pd.read_csv(self.metrics_csv, comment='#')
            if "algo" in existing.columns and "subject" in existing.columns:
                mask_to_remove = existing.apply(
                    lambda row: (row["algo"], row["subject"]) in current_combos,
                    axis=1
                )
                existing = existing[~mask_to_remove]
                combined = pd.concat([existing, self._metrics_df], ignore_index=True)
            else:
                combined = self._metrics_df
        else:
            combined = self._metrics_df
        config_line = f"# lr={self.learning_rate}, n_epoch={self.n_epoch}, batch_size={self.batch_size}, window_size={self.window_size}, overlap={self.overlap}, resample={self.resample}, imu_filter={self.imu_filter}, opensim_filter={self.opensim_filter}, y_scaling={self.y_scaling}\n"

        with open(self.metrics_csv, 'w') as f:
            f.write(config_line)
            combined.to_csv(f, index=False)
        subjects_saved = sorted(self._metrics_df["subject"].unique())
        

    def _create_weights(self, window_size):
        """Triangular weights that ramp up and down across the window.
        Middle of the window has weight 1, edges have weight 0.5."""
        half = window_size // 2
        weights = np.zeros(window_size)
        weights[:half] = np.linspace(0.5, 1.0, half)
        weights[half:] = np.linspace(1.0, 0.5, window_size - half)
        return weights
        
    
    def _overlap_mean_reconstruct(self, windows, window_size, stride):
        """
        Erstellt weighted average von predictions zu überschneidenden Zeitpunkten.
        Mittlere Frames erhalten höheres Gewicht als Randframes.
        """
        n, t, j = windows.shape #bsp 155,256,7
       
    
        total_length = (n - 1) *  stride + window_size
        signal  = np.zeros((total_length, j), dtype=np.float64)
        coverage = np.zeros((total_length, 1), dtype=np.float64)
    
        weights = self._create_weights(self.window_size)  # shape: (window_size,)
    
        for i in range(n):
            start = i *  stride
            end   = start + window_size
    
            # weights[:, None] broadcasts over j features
            signal[start:end, :]   += windows[i, :, :] * weights[:, None]
            coverage[start:end, :] += weights[:, None]
    
        # Avoid division by zero
        mask = coverage[:, 0] > 0
        signal[mask] /= coverage[mask]
    
        return signal  

    '''def _overlap_mean_reconstruct(self, windows, window_size, stride,):
        """
        Erstellt mean werte von predoctions zu überschneidenden Zeitpunkten
        """
        n, t, j = windows.shape

        total_length = (n - 1) * stride + window_size

        signal      = np.zeros((total_length, j), dtype=np.float64)
        count       = np.zeros((total_length, 1), dtype=np.float64)

        for i in range(n):
            start = i * stride
            end   = start + window_size
            signal[start:end, :] += windows[i, :, :]
            count[start:end, :]  += 1.0
        # Mean bilden 
        signal /= count

        return signal'''

    def __del__(self):
        try:
            if not self._metrics_df.empty:
                self.save()
        except Exception:
            pass