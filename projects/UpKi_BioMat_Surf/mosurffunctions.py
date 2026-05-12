import os
import glob
import re
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.signal import butter, filtfilt

# ===========================
# CONFIGURATION LOADING
# ===========================

def load_config(config_path="config/config.yaml"):
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
# SIGNAL PROCESSING UTILITIES
# ===========================

def apply_bandpass_filter(data, lowcut=0.5, highcut=20, fs=50, order=4):
    """
    Apply bandpass filter to remove DC offset and high-frequency noise.
    
    Args:
        data: numpy array [N, features]
        lowcut: Low cutoff frequency in Hz (removes gravity/drift)
        highcut: High cutoff frequency in Hz (removes sensor noise)
        fs: Sampling frequency
        order: Filter order
    
    Returns:
        filtered_data: numpy array [N, features]
    """
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')
    
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = filtfilt(b, a, data[:, i])
    
    return filtered


def apply_lowpass_filter(data, cutoff=20, fs=50, order=4):
    """
    Apply low-pass filter to remove high-frequency noise.
    
    Args:
        data: numpy array [N, features]
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency
        order: Filter order
    
    Returns:
        filtered_data: numpy array [N, features]
    """
    nyq = fs / 2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = filtfilt(b, a, data[:, i])
    
    return filtered


def compute_derived_features(accel, gyro):
    """
    Compute derived features from raw IMU signals.
    
    Features added:
    - Acceleration magnitude
    - Gyroscope magnitude
    - Jerk (derivative of acceleration)
    - Angular acceleration (derivative of gyroscope)
    
    Args:
        accel: [N, 3] acceleration in m/s²
        gyro: [N, 3] angular velocity in rad/s
    
    Returns:
        features: [N, 13] - original 6 + 7 derived features
    """
    # Magnitudes
    acc_mag = np.linalg.norm(accel, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)
    
    # Jerk (rate of change of acceleration)
    jerk = np.diff(accel, axis=0, prepend=accel[0:1])
    jerk_mag = np.linalg.norm(jerk, axis=1, keepdims=True)
    
    # Angular acceleration
    ang_accel = np.diff(gyro, axis=0, prepend=gyro[0:1])
    ang_accel_mag = np.linalg.norm(ang_accel, axis=1, keepdims=True)
    
    # Combine all features: 3 acc + 1 acc_mag + 3 gyro + 1 gyro_mag + 3 jerk + 1 jerk_mag + 1 ang_accel_mag + 1 = 14
    return np.concatenate([
        accel,           # 3 - original acceleration
        acc_mag,         # 1 - acceleration magnitude
        gyro,            # 3 - original gyroscope
        gyro_mag,        # 1 - gyroscope magnitude
        jerk,            # 3 - jerk (derivative of acceleration)
        jerk_mag,        # 1 - jerk magnitude
        ang_accel_mag,   # 1 - angular acceleration magnitude    
    ], axis=1)


# ===========================
# DATA LOADING FUNCTIONS
# ===========================

def flatten(all_files, drop=[]):
    """Flatten nested list of file paths."""
    return [item for sublist in all_files if sublist not in drop for item in sublist]


def create_mosurf_dataframes(files, activity_list, downsample, resample_activities, synthetic=False):
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


def read_mosurf(participant_list, path_raw, activity_list, downsample, resample_activities):
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
            valid_paths, activity_list, downsample, resample_activities, synthetic=False
        )
    
    return participants


def create_synthetic_dataframes(files, activity_list, downsample):
    """
    Create dataframes from synthetic IMU CSV files.
    Handles flexible naming (e.g., walking.csv vs walking_combined.csv)
    
    Args:
        files: List of file paths
        activity_list: List of activity names
        downsample: Downsampling factor
    
    Returns:
        [data_x, data_y]: Features and labels
    """
    data_frames = []
    
    # Create mapping for flexible matching
    activity_map = {}
    for idx, activity in enumerate(activity_list):
        activity_map[activity] = idx
        # Also add alternate names
    
    for path in files:
        filename = os.path.basename(path)
        
        # Try direct match first
        if filename in activity_map:
            idx = activity_map[filename]
        else:
            # Skip files that don't match any activity
            continue
        
        try:
            df = pd.read_csv(path)  # Read the synthetic CSV file
            
            # Downsample
            df = df.iloc[::downsample, :].reset_index(drop=True)
            num_rows = len(df)
            
            # Reset time column to start at 0
            if downsample == 2:  # 50Hz
                df["Time,s"] = np.round(np.arange(0, num_rows * 0.02, 0.02), 2)[:num_rows]
            elif downsample == 1:  # 100Hz
                df["Time,s"] = np.round(np.arange(0, num_rows * 0.01, 0.01), 2)[:num_rows]
            
            df["synthetic"] = 1  # Flag as synthetic data
            df["activityID"] = idx  # Assign activity ID
            
            # Reorder columns to make "Time,s" the first column
            cols = ['Time,s'] + [col for col in df.columns if col != 'Time,s']
            df = df[cols]
            
            data_frames.append(df)
        except Exception as e:
            print(f"  Warning: Could not read {filename}: {str(e)[:50]}")
            continue
    
    if not data_frames:
        print(f"  Warning: No valid synthetic data found")
        return [pd.DataFrame(), pd.Series(dtype=int)]
    
    dataset_frame = pd.concat(data_frames, ignore_index=True)
    data_x = dataset_frame.iloc[:, :-2]  # All columns except activityID and synthetic
    data_y = dataset_frame.iloc[:, -1]   # activityID
    return [data_x, data_y]


def read_synthetic(participant_list, subfolder_number, synthetic_path):
    """
    Read synthetic IMU data for specified participants.
    
    Args:
        participant_list: List of participant IDs
        subfolder_number: Subfolder index (0-8 for 9 variations)
        synthetic_path: Root path to synthetic data
    
    Returns:
        participants: Dict {participant_id: [data_x, data_y]}
    """
    participants = {}
    
    for entry in os.scandir(synthetic_path):
        if os.path.basename(entry.name) not in participant_list:
            continue
        
        valid_paths = []
        
        # Specify the subfolder number you want to process (e.g., 0)
        subfolder = os.path.join(entry.path, str(subfolder_number))
        
        if not os.path.exists(subfolder):
            print(f"Warning: Subfolder {subfolder_number} not found for {entry.name}")
            continue
        
        # Check the files in the specified subfolder
        for file in os.listdir(subfolder):
            if file.endswith(".csv") or file.endswith(".CSV"):
                file_path = os.path.join(subfolder, file)
                if os.path.basename(file) in activity_list:
                    valid_paths.append(file_path)
        
        participants[entry.name] = create_synthetic_dataframes(valid_paths, activity_list, downsample)
        
        print(f"Data read from subfolder {subfolder_number} for participant {entry.name} with {len(valid_paths)} files.")
    
    return participants


# ===========================
# MOTION CAPTURE FUNCTIONS
# ===========================

def get_mocap_path(participant, mosurf_path):
    """Get path to MoCap results for a participant."""
    return os.path.join(mosurf_path, participant, "MoCap", "ikResults_spine")


def read_mocap(participant_list, mosurf_path):
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
        mocap_path = get_mocap_path(participant, mosurf_path)

        if not os.path.exists(mocap_path):  
            print(f"Warning: MoCap folder not found for {participant}")
            continue

        joint_angle_files = glob.glob(os.path.join(mocap_path, "*.mot"))

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


# ===========================
# FEATURE EXTRACTION
# ===========================

def get_imu_placements(joint_angle, joint_to_imu_lookup):
    """
    Retrieves the two IMU placements for a given joint angle.

    Args:
        joint_angle: The name of the joint angle
        joint_to_imu_lookup: Dictionary mapping joint angles to IMU pairs

    Returns:
        tuple: A tuple containing the two corresponding IMU placements, or None
    """
    return joint_to_imu_lookup.get(joint_angle, None)


def align_imu_and_mocap(imu_df, mocap_df):
    """
    Aligns IMU and MoCap dataframes based on rounded timestamps.
    
    Args:
        imu_df: IMU dataframe with 'Time,s' column
        mocap_df: MoCap dataframe with 'time' column
    
    Returns:
        synced_df: Synchronized dataframe
    """
    # Noraxon often uses 'Time,s', MoCap often uses 'time'
    if 'time' in mocap_df.columns:
        mocap_df = mocap_df.rename(columns={'time': 'Time,s'})
    
    # Round to 3 decimal places (1ms precision) to fix floating point mismatch
    imu_df['Time,s'] = imu_df['Time,s'].round(3)
    mocap_df['Time,s'] = mocap_df['Time,s'].round(3)
    
    # Inner join keeps only the timestamps present in BOTH systems
    synced_df = pd.merge(imu_df, mocap_df, on='Time,s', how='inner')
    
    # Check synchronization quality
    sync_quality = len(synced_df) / min(len(imu_df), len(mocap_df)) * 100
    if sync_quality < 90:
        print(f"Warning: Low sync quality: {sync_quality:.1f}% of samples retained")
    
    return synced_df


# ===========================
# INITIALIZATION
# ===========================

# Load config
config = load_config()
data_cfg = config["data"]
downsample = data_cfg["downsample"]
participant_list = data_cfg["participant_list"]
activity_list = data_cfg["activity_list"]
mosurf_path = data_cfg["mosurf_path"]
resample_activities = []
synthetic_path = data_cfg["synthetic_path"]

# Load IMU and MoCap data
print("Loading IMU data")
participants = read_mosurf(participant_list, mosurf_path, activity_list, downsample, resample_activities)
print("Loading MoCap data")
mocap_participants = read_mocap(participant_list, mosurf_path)

# Load lookup table from CSV
csv_path = data_cfg.get("lookup_csv_path", "joint_imu_lookup.csv")
if not os.path.exists(csv_path):
    # Try alternate locations
    possible_paths = [
        "joint_imu_lookup.csv",
        "/home/traju/loki/joint_imu_lookup.csv",
        "/Users/thejaswini/Desktop/studyproject/ma_project_thejaswini/joint_imu_lookup.csv"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break

lookup_df = pd.read_csv(csv_path)
joint_to_imu_lookup = lookup_df.set_index("Joint_Angle")[["IMU_1", "IMU_2"]].to_dict(orient="index")

print("Preprocessing initialization complete.")