from utils.read_write import read_csv
import os
import pickle
from config import get_config_universal

#liest Daten ein

class DataReader:
    def __init__(self, config, subject_list, activity_list):
        self.config = config
        self.subject_list = subject_list
        self.activity_list = activity_list

    #liest csv Daten aus einem bestimmten directory 'conditions' ein
    #fügt die daten einer metadata Liste hinzu
    def read_metadata(self):
        metadata = []
        for subject in self.subject_list:
            subject_dir_temp = config['dataset_path'] + subject + '/'
            subject_dir = subject_dir_temp + os.listdir(subject_dir_temp)[0]
            for activity in activity_list:
                for (root, dirs, files) in os.walk(subject_dir + '/' + activity + '/conditions/'):
                    for name in files:
                        if name[-4:] == '.csv':
                            file = os.path.join(root, name)
                            metadata.append(read_csv(file))
        return metadata

    #liest csv Daten aus einem den anderen directorys hinzu
    #fügt die daten einer data Liste hinzu
    def read_data(self, data_type):
        data = []
        for subject in self.subject_list:
            subject_dir_temp = config['dataset_path'] + subject + '/'
            subject_dir = subject_dir_temp + os.listdir(subject_dir_temp)[0]
            for activity in activity_list:
                for (root, dirs, files) in os.walk(subject_dir + '/' + activity + '/' + data_type + '/'):
                    for name in files:
                        if name[-4:] == '.csv':
                            file = os.path.join(root, name)
                            data.append(read_csv(file))
        return data

#definitionen: spezifisch für Datenset
config = get_config_universal('camargo')
activity_list = ['levelground', 'ramp', 'stair']
datatype_list = ['imu', 'ik']
subject_list = ["AB06", "AB07", "AB08", "AB09", "AB10", "AB11", "AB12", "AB13", "AB14", "AB15", "AB16", "AB17", "AB18",
"AB19", "AB20", "AB21", "AB23", "AB24", "AB25"]

datareader_handler = DataReader(config, subject_list, activity_list)
metadata = datareader_handler.read_metadata()
dl_dataset = config['dl_dataset_path']
dataset_file = dl_dataset + "".join(activity_list) + '_' + "".join(datatype_list) + '_' + "".join(subject_list) + '.p'
dataset = {}
dataset['metadata'] = metadata
dataset['dataset_info'] = [('dataset_name', 'camargo'),
                           ('activity_list', activity_list),
                           ('subject_list', subject_list),
                           ('datatype_list', datatype_list)]

for datatype in datatype_list:
    data = datareader_handler.read_data(datatype)
    dataset[datatype] = data

if os.path.isfile(dataset_file):
    print('file exist')
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
else:
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)





