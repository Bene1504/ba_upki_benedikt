import sys
import json
import yaml
sys.path.append('/home/benedikt_nothhelfer/projects/BioMAT')
from loading.loadh5dataset import LoadH5DataSet
from config import get_config_universal


config = {
    'dataset_name': 'ultramocap',
    'dl_dataset_path': '/home/benedikt_nothhelfer/data_share/ULTra-MoCap/ULTra-MoCap-processed',
    'selected_sensors': ['1', '2', '3', '4', '5', '6'],
    'selected_imu_features': ['ACCX', 'ACCY', 'ACCZ', 'GYROX', 'GYROY', 'GYROZ'],
    'selected_opensim_labels': ['arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 'pro_sup_r'],
    "selected_trial_type": ["AS", "CB", "EF", "ER", "OR"]
}
config = get_config_universal('ultramocap')

# Loader testen
loader = LoadH5DataSet(config)
dataset = loader.load_dataset()

# Ergebnisse prüfen
print(f"Anzahl Trials: {len(dataset['imu'])}")
print(f"IMU Shape (erster Trial): {dataset['imu'][0].shape}")
print(f"IK Shape (erster Trial): {dataset['ik'][0].shape}")
print(f"IMU Spalten: {dataset['imu'][0].columns.tolist()}")
print(f"IK Spalten: {dataset['ik'][0].columns.tolist()}")

# Wertebereiche prüfen (nach Einheiten-Umrechnung)
imu = dataset['imu'][0]
print(f"\nIMU Wertebereiche (nach Umrechnung):")
for col in imu.columns[:6]:  # Erste 6 Spalten
    print(f"  {col}: min={imu[col].min():.3f}, max={imu[col].max():.3f}")