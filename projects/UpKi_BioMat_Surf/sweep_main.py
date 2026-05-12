import torch
import numpy as np
import os
import wandb
from sklearn.linear_model import LinearRegression
from torch.utils.data.dataloader import DataLoader
from config import get_config_universal, get_sweep_config_universal
from dataset import DataSet
from datasetbuilder import DataSetBuilder
from parametertuning import ParameterTuning
from train import Train
from test import Test
from evaluation import Evaluation
from utils.update_config import update_config
from utils.utils import get_activity_index_test, get_model_name_from_activites
from visualization.wandb_plot import wandb_plot_true_pred, wandb_plotly_true_pred
from main_universal import run_lopo
from datetime import datetime

def run_main(model_name):
    torch.manual_seed(0)
    np.random.seed(42)

    config = get_config_universal('mosurf')
    sweep_config = get_sweep_config_universal(f'mosurf_{model_name.replace("hernandez2021cnnlstm", "cnnlstm").replace("transformertsai", "transformer")}')
    default_config = config
    wandb.init(config=default_config)
    wandb_config = wandb.config
    config = update_config(config, sweep_config, wandb_config)
    config['save_model'] = False
    config['training_style'] = 'lopo'
    config['model_name'] = model_name
    config['sweep_mode'] = True

    wandb.run.name = f"{model_name}_{datetime.now().strftime('%d.%m')}"
    Evaluation.setup_wandb_metrics(config['selected_opensim_labels'])

    try:
        run_lopo(config)
    except Exception as e:
        print(f"Run failed: {e}")
        raise
    finally:
        torch.cuda.empty_cache()
        wandb.finish()

if __name__ == '__main__':
    run_main()


