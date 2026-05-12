import glob
import re
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.signal import butter, filtfilt
from utils.read_write import read_csv
import os
import pickle
from config import get_config_universal

        
def flatten(all_files, drop=[]):
    """Flatten nested list of file paths."""
    return [item for sublist in all_files if sublist not in drop for item in sublist]

def create_mosurf_dataframes(files, activity_list, downsample, synthetic=False):
    """
    Create dataframes from MoSurf IMU CSV files.
    
    Args:
    files: List of file paths
    activity_list: List of activity names to process
    downsample: Downsampling factor
    resample_activities: Activities requiring resampling (legacy, kept for compatibility)
    synthetic: Flag indicating synthetic data
    
    Returns:
    [data_x, data_y]: Features and labels
    """
    data_frames = []
    for idx, activity in enumerate(activity_list):
        for path in files:
            if os.path.basename(path) != activity:
                continue
          
            df = pd.read_csv(path, skiprows=4)  # Remove headers            
            # Remove rows where sync is 0
            df = df[df["Noraxon MyoMotion.Sync,On"] == 1]
            df.columns = df.columns.str.strip().str.rstrip(',')
            
            # Remove Mag values (we don't use magnetometer)
            df.drop(columns=[col for col in df.columns if "Mag" in col], inplace=True)
            
            # Downsample
            df = df.iloc[::downsample, :].reset_index(drop=True)
            
            df["synthetic"] = 1 if synthetic else 0
            df["activityID"] = idx  # Assign activity ID
            data_frames.append(df)
    
    dataset_frame = pd.concat(data_frames, ignore_index=True)
    
    data_x = dataset_frame.iloc[:, :-2]  # All data except synthetic and activityID
    data_y = dataset_frame.iloc[:, -1]   # activity ID column
    return [data_x, data_y]

def read_mosurf( participant_list, path_raw, activity_list, downsample):
    """
        Read MoSurf dataset for specified participants.
    
        Args:
        participant_list: List of participant IDs
        path_raw: Root path to MoSurf data
        activity_list: List of activities to load
        downsample: Downsampling factor
        resample_activities: Activities requiring resampling
    
        Returns:
            participants: Dict {participant_id: [data_x, data_y]}
    """
    participants = {}
    
    for entry in os.scandir(path_raw):  # Iterate over each participant folder
        if os.path.basename(entry) not in participant_list:
            continue
        
        all_files = []
        valid_paths = []
        subfolder = os.path.join(entry.path, "IMU")
        
        if not os.path.exists(subfolder):
            print(f"Warning: IMU folder not found for {os.path.basename(entry)}")
            continue
            
        for activity_folder in os.scandir(subfolder):
            all_files.append(glob.glob(os.path.join(activity_folder.path, "*.csv")))
        
        all_files = flatten(all_files, drop=[[]])
        for file in all_files:
            if os.path.basename(file) in activity_list:
                valid_paths.append(file)
        
        print(f"Number of valid paths for {os.path.basename(entry)}: {len(valid_paths)}")
        participants[os.path.basename(entry)] = create_mosurf_dataframes(
            valid_paths, activity_list, downsample, synthetic=False
        )
    
    return participants
    
    
def read_mocap( participant_list, dataset_path):
    """
        Read OpenSim inverse kinematics results (joint angles).
    
        Args:
            participant_list: List of participant IDs
            mosurf_path: Root path to MoSurf data
    
        Returns:
            mocap_data: Dict {participant: {activity: joint_angles_df}}
    """
    mocap_data = {}

    for participant in participant_list:
        mocap_path = os.path.join(dataset_path, participant, "MoCap", "ikResults_spine")

        if not os.path.exists(mocap_path):  
            print(f"Warning: MoCap folder not found for {participant}")
            continue

        joint_angle_files = glob.glob(os.path.join(mocap_path, "*neu.mot"))

        participant_angles = {}
        for file in joint_angle_files:
            file_name = os.path.basename(file)
            participant_id = file_name.split('_')[0]
            match = re.match(f"^{participant_id}_(.*?)_IK_results_sync_spine(?:_(\d+))?_neu.mot$", file_name)

            if match:
                base_activity = match.group(1)
                number = match.group(2)
                activity_name = f"{base_activity}_{number}" if number else base_activity
            
                participant_angles[activity_name] = extract_joint_angles(file)

        mocap_data[participant] = participant_angles
    
    return mocap_data
    
def extract_joint_angles(file_path):
    """
        Extract joint angles from OpenSim .mot file.
    
        Args:
            file_path: Path to .mot file
    
        Returns:
            df: DataFrame with time and joint angles
        """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find where the actual data starts
    for i, line in enumerate(lines):
        if "endheader" in line.lower():
            start_index = i + 1
            break

    # Read data into a DataFrame
    df = pd.read_csv(file_path, sep='\s+', skiprows=start_index)

    return df

def load_config(config_path="configs/config.yaml"):
    """Find and load a YAML config file.

        This searches upwards from the location of this file for the given
        relative `config_path` (default: 'config/config.yaml'). If not found
        it will also try the current working directory. If still not found,
        a FileNotFoundError is raised listing attempted paths.

        Args:
            config_path: path (absolute or relative) to the config YAML file.
        Returns:
            Parsed YAML as Python object (via yaml.safe_load).
        """
    cfg_rel = Path(config_path)

    # Start searching from this file's directory and go upwards.
    start = Path(__file__).resolve().parent
    attempted = []
    p = start
    while True:
        candidate = (p / cfg_rel).resolve()
        attempted.append(str(candidate))
        if candidate.exists():
            with candidate.open("r") as f:
                return yaml.safe_load(f)
        if p == p.parent:
            break
        p = p.parent

    # Fallback: try current working directory
    cwd_candidate = (Path.cwd() / cfg_rel).resolve()
    attempted.append(str(cwd_candidate))
    if cwd_candidate.exists():
        with cwd_candidate.open("r") as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Config file not found. Tried the following locations:\n" + "\n".join(attempted)
    )


# ===========================
# INITIALIZATION
# ===========================

# Load config
configjson = get_config_universal('mosurf')
config = load_config()
data_cfg = config["data"]
downsample = data_cfg["downsample"]
participant_list = configjson["lopo_subjects"]
activity_list_csv = configjson["activity_list_csv"]
activity_names = configjson["activity_list"]
mosurf_path = configjson['dataset_path']
selected_angles =configjson["selected_opensim_labels"]
#will create a picklefile to load data faster
output_file = os.path.join(configjson['dl_dataset_path'],configjson['dl_dataset'])


#Erstelle einheitliches Naming von Spaltennamen
sensor_map  = configjson['noraxon_sensor_mapping']
feature_map = configjson['noraxon_feature_mapping']
selected_imu_cols = []
imu_col_names     = []
for sensor in configjson['selected_sensors']:
    for feature in configjson['selected_imu_features']:
        selected_imu_cols.append(f"{sensor_map[sensor]} {feature_map[feature]}")
        imu_col_names.append(f"{sensor}_{feature}")



# Load IMU and MoCap data
if os.path.isfile(output_file):
    print(f'File already exists: {output_file}')
else:
    print("Loading IMU data")
    participants_imu = read_mosurf(participant_list, mosurf_path, activity_list_csv, downsample=1)
    print("Loading MoCap data")
    participants_mocap = read_mocap(participant_list, mosurf_path)

    imu_list      = []
    ik_list       = []
    metadata_list = []

    for participant in participant_list:
        if participant not in participants_imu:
            print(f'Skipping {participant}: no IMU data')
            continue
        if participant not in participants_mocap:
            print(f'Skipping {participant}: no MoCap data')
            continue

        imu_full_df, activity_ids = participants_imu[participant]
        mocap_activities = participants_mocap[participant]

        for act_idx, (act_csv, act_name) in enumerate(zip(activity_list_csv, activity_names)):
            if act_name not in mocap_activities:
                print(f'  {participant} – no MoCap for {act_name}, skipping')
                continue

            mocap_df = mocap_activities[act_name].copy()

            # filter IMU nach activity
            mask = (activity_ids == act_idx).values
            imu_act = imu_full_df[mask].reset_index(drop=True)

            if len(imu_act) == 0:
                print(f'  {participant} {act_name}: no IMU samples, skipping')
                continue

            # Längen prüfen
            if abs(len(imu_act) != len(mocap_df)) > 1:
                print(f'  Warning {participant} {act_name}: IMU {len(imu_act)} != MoCap {len(mocap_df)} rows, skipping')
                continue
            # Auf gleiche Länge kürzen falls Zeilenunterschied von maximal 1
            min_len = min(len(imu_act), len(mocap_df))
            imu_act = imu_act.iloc[:min_len].reset_index(drop=True)
            mocap_df = mocap_df.iloc[:min_len].reset_index(drop=True)

            # IMU extrahieren
            missing_imu = [c for c in selected_imu_cols if c not in imu_act.columns]
            if missing_imu:
                print(f'  Warning {participant} {act_name}: missing IMU cols: {missing_imu}')
                continue

            imu_df = imu_act[selected_imu_cols].copy()
            imu_df.columns = imu_col_names

            #Einheiten konvertieren
            acc_cols  = [c for c in imu_col_names if "ACC"  in c]
            gyro_cols = [c for c in imu_col_names if "GYRO" in c]
            imu_df[acc_cols]  = imu_df[acc_cols]  / 1000.0
            imu_df[gyro_cols] = imu_df[gyro_cols] * (np.pi / 180.0)

            imu_df = imu_df.reset_index(drop=True)

            # Winkel extrahieren
            missing_angles = [a for a in selected_angles if a not in mocap_df.columns]
            if missing_angles:
                print(f'  Warning {participant} {act_name}: missing angles: {missing_angles}')
                continue

            ik_df = mocap_df[selected_angles].copy().reset_index(drop=True)

            # NaN entfernen
            nan_mask = ~(imu_df.isna().any(axis=1) | ik_df.isna().any(axis=1))
            imu_df = imu_df[nan_mask].reset_index(drop=True)
            ik_df  = ik_df[nan_mask].reset_index(drop=True)

            # Metadata
            meta_df = pd.DataFrame({
                'subject':   [participant] * len(imu_df),
                'movement':  [act_name]   * len(imu_df),
                'trialType': [act_name]   * len(imu_df),
                'time':      (np.arange(len(imu_df)) * 0.01).round(3)
            })

            imu_list.append(imu_df)
            ik_list.append(ik_df)
            metadata_list.append(meta_df)
            
    print(f'\nTotal trials: {len(imu_list)}')

    dataset = {
        'imu':      imu_list,
        'ik':       ik_list,
        'metadata': metadata_list,
        'dataset_info': [
            ('dataset_name',     'mosurf'),
            ('participant_list', participant_list),
            ('activity_list',    activity_names),
            ('selected_angles',  selected_angles),
        ]
    }

    os.makedirs(configjson['dl_dataset_path'], exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved to {output_file}')        