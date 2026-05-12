import os
import pickle
from config import get_config_universal
from loading.loadh5dataset import LoadH5DataSet

config = get_config_universal('ultramocap')

output_file = os.path.join(config['dl_dataset_path'], 'ultramocap_dataset.p')

if os.path.isfile(output_file):
    print('File already exists, skipping.')
else:
    print('Loading from H5...')
    loader = LoadH5DataSet(config)
    imu_data, ik_data, metadata = loader.run_get_dataset()
    
    dataset = {
        'imu': imu_data,
        'ik': ik_data,
        'metadata': metadata,
        'dataset_info': [('dataset_name', 'ultramocap')]
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f'Saved to {output_file}')