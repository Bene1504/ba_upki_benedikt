"""
Boxplot plots for hyperparameter sweep analysis.
One boxplot per hyperparameter, showing effect on lopo_RMSE_overall_mean.
Separated by architecture (hue) and dataset (separate figures).

Run: python plot_boxplots.py
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Output 
OUT_DIR = "/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Surf/results/boxplots"
os.makedirs(OUT_DIR, exist_ok=True)

# Style
COLORS = {"Transformer": "#1f78b4", "BiLSTM": "#e31a1c", "CNN-LSTM": "#33a02c"}
plt.rcParams.update({
    "font.family":        "serif",
    "font.weight":        "bold",
    "axes.labelweight":   "bold",
    "axes.titleweight":   "bold",
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
})

#File paths 
MOSURF_FILES = [
    "/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Surf/wandb_export_2026-04-20T14_31_06.859+02_00.csv",   # Transformer (20)
    "/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Surf/wandb_export_2026-04-20T14_41_28.004+02_00.csv",   # CNN-LSTM (40)
    "/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Surf/wandb_export_2026-04-20T14_42_11.525+02_00.csv",   # BiLSTM (40)
]
ULTRA_FILES = [
    "/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Surf/ultra_wandb_export_2026-04-20T14_47_15.619+02_00.csv",  # Transformer
    "/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Surf/ultra_wandb_export_2026-04-20T14_46_57.522+02_00.csv",  # BiLSTM
    "/home/benedikt_nothhelfer/ba_upki_benedikt/projects/UpKi_BioMat_Surf/ultra_wandb_export_2026-04-20T14_47_45.741+02_00.csv",  # CNN-LSTM
]


def detect_arch(name):
    name = str(name).lower()
    if "bilstm" in name:
        return "BiLSTM"
    elif "cnnlstm" in name or "hernandez" in name:
        return "CNN-LSTM"
    elif "transformer" in name or "tsai" in name:
        return "Transformer"
    return "Unknown"

def load_files(file_list):
    dfs = []
    for f in file_list:
        df = pd.read_csv(f)
        df["arch"] = df["Name"].apply(detect_arch)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df_mosurf = load_files(MOSURF_FILES)
df_ultra  = load_files(ULTRA_FILES)

# Hyperparameter groups 
# (col, display_name, archs_it_applies_to)
SHARED_PARAMS = [
    ("learning_rate",   "Learning Rate",  ["Transformer", "BiLSTM", "CNN-LSTM"]),
    ("batch_size",      "Batch Size",     ["Transformer", "BiLSTM", "CNN-LSTM"]),
    ("n_epoch",         "Epochs",         ["Transformer", "BiLSTM", "CNN-LSTM"]),
    ("l2_weight_decay", "L2 Weight Decay",["Transformer", "BiLSTM", "CNN-LSTM"]),
]
BILSTM_PARAMS = [
    ("bilstm_dropout_p",   "Dropout",      ["BiLSTM"]),
    ("bilstm_hidden_size", "Hidden Size",  ["BiLSTM"]),
    ("bilstm_num_layers",  "Num Layers",   ["BiLSTM"]),
]
CNNLSTM_PARAMS = [
    ("DecoderLSTM_dropout_p",        "LSTM Dropout",       ["CNN-LSTM"]),
    ("DecoderLSTM_hidden_size",      "LSTM Hidden Size",   ["CNN-LSTM"]),
    ("DecoderLSTM_num_layers",       "LSTM Num Layers",    ["CNN-LSTM"]),
    ("EncoderCNN_conv_l1_out_channel","Conv L1 Channels",  ["CNN-LSTM"]),
    ("EncoderCNN_conv_l2_out_channel","Conv L2 Channels",  ["CNN-LSTM"]),
]
TRANSFORMER_PARAMS = [
    ("tsai_d_ff",          "Feed-Forward Dim",  ["Transformer"]),
    ("tsai_d_model",       "Model Dim",         ["Transformer"]),
    ("tsai_fc_dropout_p",  "FC Dropout",        ["Transformer"]),
    ("tsai_res_dropout_p", "Res Dropout",       ["Transformer"]),
    ("tsai_n_heads",       "Num Heads",         ["Transformer"]),
    ("tsai_n_layers",      "Num Layers",        ["Transformer"]),
]

ALL_PARAMS = SHARED_PARAMS + BILSTM_PARAMS + CNNLSTM_PARAMS + TRANSFORMER_PARAMS

METRIC_COL  = "lopo_RMSE_overall_mean"
METRIC_LABEL = "RMSE (°)"

#Plot function
def make_boxplot(df, col, display_name, archs, dataset_name, outpath):
    subset = df[df["arch"].isin(archs)].copy()
    subset = subset.dropna(subset=[col, METRIC_COL])
    if subset.empty:
        print(f"Skipping {col} for {dataset_name} — no data")
        return

    # Cast to string so seaborn treats as categorical
    subset[col] = subset[col].astype(str)

    # Sort x-axis values numerically if possible
    try:
        order = sorted(subset[col].unique(), key=lambda v: float(v))
    except ValueError:
        order = sorted(subset[col].unique())

    palette = {a: COLORS[a] for a in archs if a in COLORS}
    hue = "arch" if len(archs) > 1 else None
    palette_arg = palette if hue else None

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.boxplot(
        data=subset,
        x=col, y=METRIC_COL,
        hue=hue,
        order=order,
        palette=palette_arg,
        ax=ax,
        linewidth=1.2,
        showfliers=False,
    )

    ax.set_xlabel(display_name, fontweight="bold")
    ax.set_ylabel(METRIC_LABEL, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    if hue:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, frameon=False,
                  loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.tight_layout()
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# Generate plots
for dataset_name, df in [("mosurf", df_mosurf), ("ultra", df_ultra)]:
    for col, display_name, archs in ALL_PARAMS:
        if col not in df.columns:
            continue
        safe_col = col.lower().replace(" ", "_").replace("/", "_")
        arch_tag = archs[0].lower().replace("-", "") if len(archs) == 1 else "shared"
        outpath  = os.path.join(OUT_DIR, f"boxplot_{dataset_name}_{arch_tag}_{safe_col}.svg")
        make_boxplot(df, col, display_name, archs, dataset_name, outpath)