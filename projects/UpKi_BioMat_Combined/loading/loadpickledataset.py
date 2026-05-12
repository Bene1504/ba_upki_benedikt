import os
import pickle
import pandas as pd


class LoadPickleDataSet:
    def __init__(self, config):
        '''
        This class loads and processes a dataset from a pickle file. The class takes a configuration dictionary as input,
        which contains various parameters such as the file path, the name of the dataset, and the selected sensors, selected_imu_features, and selected_opensim_labels.
        :param config:
        '''
        self.dl_dataset_path = config['dl_dataset_path']
        self.dataset_name = config['dl_dataset']
        self.selected_sensors = config['selected_sensors']
        self.selected_imu_features = config['selected_imu_features']
        self.selected_opensim_labels = config['selected_opensim_labels']
        self.dataset = []

    def load_dataset(self):
        dataset_file = self.dl_dataset_path + self.dataset_name
        if os.path.isfile(dataset_file):
            print('file exist')
            # rb = read the file as binary data
            with open(dataset_file, 'rb') as f:
                self.dataset = pickle.load(f)
                
        else:
            print('this dataset is not exist: run run_dataset_prepration.py or ultramocap_preparation.py first')

    def combine_sensors_features(self):
        self.selected_sensor_features = []
        for sensor in self.selected_sensors:
            ss = [sensor + '_' + imu_feature for imu_feature in self.selected_imu_features]
            self.selected_sensor_features = self.selected_sensor_features + ss

    def get_selected_ik(self):
        ik = self.dataset['ik']
        self.ik = [y_val[self.selected_opensim_labels].values for i, y_val in enumerate(ik)]
        return ik

    def get_selected_imu(self):
        imu = self.dataset['imu']
        self.combine_sensors_features()
        self.imu = [y_val[self.selected_sensor_features].values for i, y_val in enumerate(imu)]
        del imu

    def run_get_dataset(self):
        self.load_dataset()
        self.get_selected_imu()
        self.get_selected_ik()

        selected_x_values = self.imu
        selected_y_values = self.ik
        selected_labels = self.dataset['metadata']

        #imu = self.dataset['imu'][0]
        #ik = self.dataset['ik'][0]
        
        #print(ik.columns.tolist())

        # Min/Max für jede Spalte
        #for col in imu.columns:
            #print(f"{col}: min={imu[col].min():.2f}, max={imu[col].max():.2f}")
        #for col in ik.columns:
            #print(f"{col}: min={ik[col].min():.2f}, max={ik[col].max():.2f}")

        del self.dataset
        return selected_x_values, selected_y_values, selected_labels



