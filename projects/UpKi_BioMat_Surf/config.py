import json
import yaml
import os

def get_config_universal(dataset_name, config_path=None):
    config_dir = os.path.dirname(os.path.abspath(__file__))
    path = config_path if config_path else os.path.join(config_dir, 'configs', f'{dataset_name}_config.json')
    with open(path) as f:
        config = json.load(f)
    #with open(os.path.join(config_dir, 'configs', f'{dataset_name}_config.json')) as f:
        #config = json.load(f)
    return config


def get_sweep_config_universal(dataset_name):
    config_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(config_dir, 'configs', f'sweep_{dataset_name}_config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_model_config(model_config):
    with open(f'./configs/{model_config}.json') as f:
        config = json.load(f)
    return config
