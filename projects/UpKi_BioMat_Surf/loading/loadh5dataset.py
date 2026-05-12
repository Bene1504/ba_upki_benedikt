import h5py
import numpy as np
import pandas as pd
import os

class LoadH5DataSet:
    def __init__(self, config):
        self.config = config
        self.dataset_path = config['dataset_path']
        self.dataset_name = config['dataset_name']
        self.selected_sensors = config['selected_sensors']
        self.selected_imu_features = config['selected_imu_features']
        self.selected_opensim_labels = config['selected_opensim_labels']
        
    def load_dataset(self):
        h5_file = os.path.join(self.dataset_path, 'All_subjects_data.h5')
        
        with h5py.File(h5_file, 'r') as f:
            # Spaltennamen extrahieren
            
            
            imu_data = []
            ik_data = []
            metadata = []
            
            for subject in sorted(f.keys()):
                for movement in f[subject].keys():
                    for speed in f[subject][movement].keys():
                        # Daten laden und zu float konvertieren
                        raw_data = f[f'{subject}/{movement}/{speed}'][:]
                        column_names = list(f[f'{subject}/{movement}/{speed}'].attrs['column_names'])

                        # ersetze leere Strings durch float
                        data = np.where(raw_data == b'', np.nan, raw_data)
                        data = data.astype(float)

                        
                        # IMU-Spalten extrahieren
                        imu_cols = self._get_imu_columns(column_names)
                        
                        imu = data[:, imu_cols]
                        
                        # Einheiten umrechnen
                        imu = self._convert_units(imu, column_names, imu_cols)
                        
                        # Gelenkwinkel extrahieren
                        ik_cols = self._get_ik_columns(column_names)
                        ik = data[:, ik_cols]
                        
                        # Metadata erstellen
                        meta = pd.DataFrame({
                            'subject': [subject] * len(data),
                            'movement': [movement] * len(data),
                            'speed': [speed] * len(data),
                            'time': data[:, 0]
                        })
                        # Gemeinsame NaN-Maske für IMU und IK um gleiche länge von IMU und IK sicherzustellen
                        nan_mask = ~(np.isnan(imu).any(axis=1) | np.isnan(ik).any(axis=1))

                        #imu_data.append(imu[nan_mask])
                        #ik_data.append(ik[nan_mask])
                        #Umstellung für die pickle datei
                        imu_col_names = self._get_imu_col_names()
                        ik_col_names = self.selected_opensim_labels

                        imu_data.append(pd.DataFrame(imu[nan_mask], columns=imu_col_names))
                        ik_data.append(pd.DataFrame(ik[nan_mask], columns=ik_col_names))
                        metadata.append(meta[nan_mask].reset_index(drop=True))
                        
                        
        
        return imu_data, ik_data, metadata
    
    def _get_imu_columns(self, column_names):
        """Findet Indizes der IMU-Spalten-nur Sensoren 1-6"""
        imu_cols = []
        for i, col in enumerate(column_names):
            if 'ACC' in col or 'GYRO' in col:
                prefix = 'ACC' if 'ACC' in col else 'GYRO' #identify Acc or Gyro number
                after_prefix = col.split(prefix)[1] #zb ACCX6 das X soll weg
                number = ''.join(filter(str.isdigit, after_prefix))  # '6' oder '10'
                if number in self.selected_sensors:
                    imu_cols.append(i)
        return imu_cols
    
    def _get_ik_columns(self, column_names):
        """Findet Indizes der Gelenkwinkel-Spalten"""
        ik_cols = []
        for label in self.selected_opensim_labels:
            for i, col in enumerate(column_names):
                if label in col:
                    ik_cols.append(i)
                    break
        return ik_cols
    
    def _get_imu_col_names(self):
        """Erstellt Spaltennamen für IMU-Daten"""
        col_names = []
        for sensor in self.selected_sensors:  # ['1','2','3','4','5','6']
            for feature in self.selected_imu_features:  # ['ACCX','ACCY','ACCZ','GYROX','GYROY','GYROZ']
                col_names.append(f'{sensor}_{feature}')
        return col_names
    
    def _convert_units(self, imu_data, column_names, imu_cols):
        """
        Konvertiert UltraMoCap Einheiten zu BioMAT Einheiten:
        - Accelerometer: mm/s² → g (teile durch 9810)
        - Gyroskop: °/s → rad/s (multipliziere mit π/180)
        """
        converted = imu_data.copy()
        
        for i, col_idx in enumerate(imu_cols):
            col_name = column_names[col_idx]
            if 'ACC' in col_name:
                # mm/s² → g
                converted[:, i] = imu_data[:, i] / 9810.0
            elif 'GYRO' in col_name:
                # °/s → rad/s
                converted[:, i] = imu_data[:, i] * (np.pi / 180.0)
        
        return converted
    
    def _get_subject_list(self):
        """Liste aller Subjects"""
        return [f'subject_{i}' for i in range(1, 14)]

    def run_get_dataset(self):
        
        return self.load_dataset()