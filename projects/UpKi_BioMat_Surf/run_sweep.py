import wandb
from config import get_sweep_config_universal
from sweep_main import run_main
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True,
                    choices=["transformertsai", "bilstm", "hernandez2021cnnlstm"])
parser.add_argument("--count", type=int, default=20)
args = parser.parse_args()

model_to_config = {
    "transformertsai":        "mosurf_transformer",
    "bilstm":                 "mosurf_bilstm",
    "hernandez2021cnnlstm":   "mosurf_cnnlstm"
}

model_to_project = {
    "transformertsai":        "mosurf_sweep_transformer",
    "bilstm":                 "mosurf_sweep_bilstm",
    "hernandez2021cnnlstm":   "mosurf_sweep_cnnlstm"
}

torch.cuda.empty_cache()
sweep_config = get_sweep_config_universal(model_to_config[args.model])
sweep_id = wandb.sweep(sweep_config, project=model_to_project[args.model])
wandb.agent(sweep_id, function=lambda: run_main(args.model), count=args.count)