#Modul Bedeutungen:
#Weights&Biases: Experiment Tracking Tool für Machine Learning
#loggt Metriken, Hyperparameter, Plots während des Trainings
import wandb
#PyTorch Deep Learning Framework
import torch
#Numerische Berechnungen mit Arrays
import numpy as np
#Dateisystem-Operationen
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#PyTorchs Batch-Loading Mechanismus
#Erstellt Batches aus dem Dataset
from torch.utils.data.dataloader import DataLoader
#Time Series Machine Learning Library
#Linear Regression als Baseline
from tslearn.metrics.cysax import LinearRegression
from config import get_config_universal
#BioMAT eigene Module/Klassen
from dataset import DataSet
from datasetbuilder import DataSetBuilder
from evaluation import Evaluation
from parametertuning import ParameterTuning
from test import Test
from train import Train
from utils.utils import get_activity_index_test, get_model_name_from_activites
#Plotly-basierte Visualisierungen für wandb
from visualization.wandb_plot import wandb_plotly_true_pred

from results_collector import ResultsCollector
import argparse
import json


def build_file_info(cfg):
    model_by_map={
        "hernandez2021cnnlstm": "CNNLSTM",
        "bilstm": "BiLSTM",
        "transformertsai": "Transformer"}
    model=model_by_map.get(cfg["model_name"])
    style        = cfg["training_style"]
    train_act    = "_".join(cfg["train_activity"])
    test_act     = "_".join(cfg["test_activity"])
    sensors      = "-".join(cfg["selected_sensors"])
    pad          = cfg["target_padding_length"]
    lr           = str(cfg["learning_rate"]).replace("0.", "")
    ovlp         = str(cfg["overlap"]).replace("0.", "")
    #bs           = cfg["batch_size"]
    loss         = cfg["loss"]
    imu_filt     = "imuFilt"  if cfg.get("imu_filter",     False) else "noImuFilt"
    osim_filt    = "osimFilt" if cfg.get("opensim_filter",  False) else "noOsimFilt"
    y_scal       = "yScal"    if cfg.get("y_scaling",       False) else "noYScal"
    run_idx = cfg.get("run_idx", 0)

    scaler_by_map = {
    "by_groupvar": "grpVar",
    "by_var":      "byVar",
    "by_sample":   "bySamp",
    "by_step":     "byStep",
    }
    scaler_by = scaler_by_map.get(cfg["data_transformer"]["data_transformer_by"], "unkown")
    
    #f"__train_{train_act}"
    #f"__test_{test_act}"
    # _bs{bs}
    return (
        f"{model}__{style}"
        f"__s{sensors}__pad{pad}_lr{lr}_o{ovlp}_{loss}"
        f"__{imu_filt}_{osim_filt}_{y_scal}_{scaler_by}"
        f"_run{run_idx:02d}"
    )
    


def run_main(model_name=None, run_idx=0):
    #Zufallszahlen-Generatoren
    #torch.manual_seed(0)
    #np.random.seed(42)
    #für finale runs
    torch.manual_seed(run_idx)
    np.random.seed(run_idx)
    
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--config", type=str, default=None)
    #parser.add_argument("--extra_info", type=str, default="na")
    #parser.add_argument("--model_name", type=str, default=None)  # ← neu
    #args = parser.parse_args()
    
    #config = get_config_universal('ultramocap', config_path=args.config)
    # für finalen 10runs
    if model_name == 'transformertsai':
        config = get_config_universal('ultramocap', config_path='configs/ultramocap_config_transformer.json')
    elif model_name == 'hernandez2021cnnlstm':
        config = get_config_universal('ultramocap', config_path='configs/ultramocap_config_cnnlstm.json')
    else:
        config = get_config_universal('ultramocap', config_path='configs/ultramocap_config_bilstm.json')


    # Model überschreiben falls übergeben
    if model_name is not None:
        config['model_name'] = model_name          # ← neu
    elif args.model_name is not None:
        config['model_name'] = args.model_name     # ← neu
    
    config['run_idx'] = run_idx
    extra_info = build_file_info(config)
    
    wandb.init(project='UpKi_Biomat_ULTra', name=extra_info, config=config)
    Evaluation.setup_wandb_metrics(config['selected_opensim_labels'])
    
    if config['training_style'] == 'lopo':
        run_lopo(config)
    else:
        run_standard(config)
    
    wandb.finish()  # ← wichtig! sonst überschreibt der nächste Run den aktuellen


def run_lopo(config):
    all_subjects = config['lopo_subjects']

    
    # Sammle mittleren RMSe über alle Aktivitäten pro Fold
    all_fold_rmse = []
    use_collector = not config.get('sweep_mode', False)
    collector = ResultsCollector(config) if use_collector else None

    for fold_idx, test_subject in enumerate(all_subjects):
        print(f"LOPO Fold: Test Subject = {test_subject}")

        # Train = alle außer test_subject
        config['train_subjects'] = [s for s in all_subjects if s != test_subject]
        config['test_subjects']  = [test_subject]

        # Dataset für diesen Fold
        dataset_handler = DataSet(config, load_dataset=True)
        kihadataset_train, kihadataset_test = dataset_handler.run_dataset_split_loop()


        kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'] = \
            dataset_handler.run_segmentation(
                kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels']
            )
        kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'] = \
            dataset_handler.run_segmentation(
                kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels']
            )
        
        # DataLoader
        #normalisierung wird im datasetbuilder gemacht
        train_dataset = DataSetBuilder(
            kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'], config, 
            transform_method=config['data_transformer'], scaler=None, y_scaler=None
        )
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=config['batch_size'], shuffle=True
        )
        test_dataset = DataSetBuilder(
            kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'], config,
            transform_method=config['data_transformer'], scaler=train_dataset.scaler, y_scaler=train_dataset.y_scaler
        )
        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=config['batch_size'], shuffle=False
        )
        
        

        # Modellname mit Subject-Info
        config['model_train_activity'], config['model_test_activity'] = \
            get_model_name_from_activites(config['train_activity'], config['test_activity'])
        model_file = (config['model_name'] + '_lopo_' + test_subject + '_' +
                      "".join(config['model_train_activity']) + f'_run{config["run_idx"]:02d}.pt')
        
        # Training
        config['epoch_offset'] = wandb.run.step
        training_handler = Train(config, train_dataloader=train_dataloader,
                                  test_dataloader=test_dataloader)
        model = training_handler.run_training()
        

        if config['save_model']:
            torch.save(model, os.path.join('./caches/trained_model/', model_file))

        # Testing
        test_handler = Test()
        y_pred, y_true, loss = test_handler.run_testing(config, model, test_dataloader=test_dataloader)
        y_true = y_true.detach().cpu().clone().numpy()
        y_pred = y_pred.detach().cpu().clone().numpy()

        shape = y_pred.shape 
        y_scaler = train_dataset.y_scaler
        if train_dataset.y_scaler is not None:
            print('inverse transforming y-values')
 
            if hasattr(y_scaler, 'subject_mean'):  # ← kein Import nötig
                y_pred_out = np.zeros_like(y_pred)
                y_true_out = np.zeros_like(y_true)
                for i in range(len(y_pred)):
                    y_pred_out[i] = y_scaler.inverse_transform(y_pred[i], subject_id=test_subject)
                    y_true_out[i] = y_scaler.inverse_transform(y_true[i], subject_id=test_subject)
                y_pred = y_pred_out
                y_true = y_true_out
                
                print(f"test_subject:      {test_subject}")
                print(f"subject_mean:      {y_scaler.subject_mean.get(test_subject, 'NOT FOUND')}")
                print(f"global_std:        {y_scaler.global_std}")
                print(f"y_true mean nach inverse: {y_true[:,:,4].mean():.2f}°")  # pro_sup_r
            else:
                y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, shape[2])).reshape(shape)
                y_true = y_scaler.inverse_transform(y_true.reshape(-1, shape[2])).reshape(shape)
 

        
        # Evaluation pro Aktivität
        fold_activity_rmse = []

        for activity in config['test_activity']:
            activity_index = get_activity_index_test(kihadataset_test['labels'], activity)

            if len(activity_index) == 0:
                print(f"Keine Samples für {activity} in {test_subject} – überspringe")
                continue

            wandb_plotly_true_pred(y_true[activity_index], y_pred[activity_index],config['selected_opensim_labels'],f'lopo_{test_subject}_{activity}')
            

            # Evaluation
            eval_handler = Evaluation(config=config,y_pred=y_pred[activity_index],y_true=y_true[activity_index],val_or_test=f'lopo_{test_subject}_{activity}',epoch=config['n_epoch'])
           
            if collector is not None:
                collector.log_metrics(subject=test_subject, activity=activity,rmse_all=eval_handler.rmse_all, mae_all=eval_handler.mae_all,n_rmse_all=eval_handler.n_rmse_all, r_all=eval_handler.r_all,
                joint_names=config['selected_opensim_labels'])
                collector.log_predictions(subject=test_subject, activity=activity,y_true=y_true[activity_index], y_pred=y_pred[activity_index], labels = kihadataset_test['labels'][activity_index],
                joint_names=config['selected_opensim_labels'])
            
            

            fold_activity_rmse.append(eval_handler.rmse_all)
        
        if collector is not None:
            collector.save()
        # Mittlerer RMSE über alle Aktivitäten dieses Folds (pro Gelenk)
        if fold_activity_rmse:
            fold_mean_rmse = np.array(fold_activity_rmse).mean(axis=0)  # Shape: (n_joints,)
            all_fold_rmse.append(fold_mean_rmse)
            fold_log = {'fold_RMSE_mean': float(fold_mean_rmse.mean()),'fold': fold_idx + 1}
            for i, label in enumerate(config['selected_opensim_labels']):
                fold_log[f'fold_RMSE_{label}'] = float(fold_mean_rmse[i])
        
            wandb.log(fold_log)
            print(f"Fold {test_subject} – RMSE mean über Aktivitäten: {fold_mean_rmse.mean():.4f}")
        

    #Gesamtzusammenfassung über alle Folds
    if all_fold_rmse:
        all_fold_rmse = np.array(all_fold_rmse)  # Shape: (n_folds, n_joints)

        lopo_summary = {}
        for i, label in enumerate(config['selected_opensim_labels']):
            lopo_summary[f'lopo_RMSE_mean_{label}'] = float(all_fold_rmse[:, i].mean())
            lopo_summary[f'lopo_RMSE_std_{label}']  = float(all_fold_rmse[:, i].std())

        lopo_summary['lopo_RMSE_overall_mean'] = float(all_fold_rmse.mean())
        lopo_summary['lopo_RMSE_overall_std']  = float(all_fold_rmse.mean(axis=1).std())

        print("LOPO Summary wird geloggt:", lopo_summary)
        wandb.run.summary.update(lopo_summary)

        print(f"LOPO abgeschlossen – {len(all_fold_rmse)} Folds")
        print(f"Gesamt RMSE: {all_fold_rmse.mean():.4f} ± {all_fold_rmse.mean(axis=1).std():.4f}")
        print(f"Model Architecture:",config['model_name'])


def run_standard(config):
    load_model = config['load_model']
    save_model = config['save_model']
    tuning = config['tuning']
    individual_plot = config['individual_plot']
    # build and split dataset to training and test
    #Dataset-Klasse: Hauptklasse für Datenmanagement
    dataset_handler = DataSet(config, load_dataset=True)
    #teilt Dataset in Training und Test
    #Struktur kihadataset_train:
    #x: np.ndarray
    #y: np.ndarray
    #labels: pd.DataFrame
    kihadataset_train, kihadataset_test = dataset_handler.run_dataset_split_loop()
    #sliding window segmentation
    kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'] = dataset_handler.run_segmentation(kihadataset_test['x'],
                                                                                     kihadataset_test['y'], kihadataset_test['labels'])# kein Problem mit labels
    #decide if perform Parameter Tuning
    if tuning == True:
        #löscht Test-Set(wird nicht für Training benötigt)
        del kihadataset_test
        ParameterTuning(config, kihadataset_train)
    else:
        #teilt Zeitreihen in überlappende Fenster
        #Shape ändert sich(vorher: Sequenzen unterschiedlicher Länge, nachher: fixe Window-Größe)
        kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'] = dataset_handler.run_segmentation(
            kihadataset_train['x'],
            kihadataset_train['y'], kihadataset_train['labels'])

        if config['model_name'] == 'linear':
            #Daten Reshape für Lineare Regression
            #Zweck: Flattening für Lineare Regression
            x_train = kihadataset_train['x'][:, :, :]
            y_train = kihadataset_train['y'][:, :, :]
            x_tr = np.reshape(x_train, [x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
            y_tr = np.reshape(y_train, [y_train.shape[0], y_train.shape[1]*y_train.shape[2]])
            x_test = kihadataset_test['x'][:, :, :]
            y_test = kihadataset_test['y'][:, :, :]
            x_true = np.reshape(x_test, [x_test.shape[0], x_test.shape[1] * x_test.shape[2]])
            y_true = y_test
            config['model_train_activity'], config['model_test_activity'] = get_model_name_from_activites(
                config['train_activity'], config['test_activity'])
            model_file = config['model_name'] + '_' + "".join(config['model_train_activity']) + \
                         '_' + "".join(config['model_test_activity']) + '.pt'
            if load_model and os.path.isfile('./caches/trained_model/' + model_file):
                model = torch.load(os.path.join('./caches/trained_model/', model_file))
            else:
                #fit trainiert lineare Regression (berechnet Gewichte via Least Squares)
                model = LinearRegression().fit(x_tr, y_tr)
                if save_model:
                    torch.save(model, os.path.join('./caches/trained_model/', model_file))

            y_pred = model.predict(x_true)
            y_pred = np.reshape(y_pred, [y_test.shape[0], y_test.shape[1], y_test.shape[2]])
        else:
            #DataLoader erstellen
            #erstellt Klasse für Train_dataset und test_dataset
            train_dataset = DataSetBuilder(kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'], config,
                                           transform_method=config['data_transformer'], scaler=None, y_scaler=None)
            #Reihenfolge in der Batches fürs Trainieren geladen werden solle
            #zufällig damit model sich nicht auf spezifische Reihenfolge optimiert
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
            test_dataset = DataSetBuilder(kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'],config, transform_method=config['data_transformer'],
                                            scaler=train_dataset.scaler, y_scaler=train_dataset.y_scaler) #kein Problem mit labels
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False)

            config['model_train_activity'], config['model_test_activity'] = get_model_name_from_activites(config['train_activity'],
                                                                                                          config['test_activity'])
            model_file = config['model_name'] + '_' + "".join(config['model_train_activity']) + \
                         '_' + "".join(config['model_test_activity']) + '.pt'
            #Training
            training_handler = Train(config, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
            model = training_handler.run_training()
            if save_model:
                torch.save(model, os.path.join('./caches/trained_model/', model_file))

            # Testing
            test_handler = Test()
            y_pred, y_true, loss = test_handler.run_testing(config, model, test_dataloader=test_dataloader)
            y_true = y_true.detach().cpu().clone().numpy()
            y_pred = y_pred.detach().cpu().clone().numpy()

            shape = y_pred.shape  
            y_scaler = train_dataset.y_scaler
            if train_dataset.y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, shape[2])).reshape(shape)
                y_true = y_scaler.inverse_transform(y_true.reshape(-1, shape[2])).reshape(shape)
                
            # Evaluation
            if individual_plot:
                for subject in config['test_subjects']:
                    subject_index = kihadataset_test['labels'][kihadataset_test['labels']['subject'] == subject].index.values
                    wandb_plotly_true_pred(y_true[subject_index], y_pred[subject_index], config['selected_opensim_labels'], str('test_'+ subject))
                    Evaluation(config=config, y_pred=y_pred[subject_index], y_true=y_true[subject_index], val_or_test=subject, epoch=config['n_epoch'])
        # Evaluation
        for activity in config['test_activity']:
            activity_to_evaluate = activity
            activity_index = get_activity_index_test(kihadataset_test['labels'], activity_to_evaluate)
            if len(activity_index) == 0:
                print(f"Keine Samples für {activity_to_evaluate} in Test-Daten – überspringe")
                continue

            wandb_plotly_true_pred(y_true[activity_index], y_pred[activity_index], config['selected_opensim_labels'], 'test_' + activity_to_evaluate)
            Evaluation(config=config, y_pred=y_pred[activity_index], y_true=y_true[activity_index], val_or_test='all_' + activity_to_evaluate, epoch=config['n_epoch'])
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--run_idx_start", type=int, default=0)
    args = parser.parse_args()
    
    
    models = [args.model_name] if args.model_name else [
        "hernandez2021cnnlstm",
        "bilstm",
        "transformertsai"
    ]
    
    for model_name in models:
        for run_idx in range(args.run_idx_start, 10):
            print(f"\n{'='*50}")
            print(f"Training: {model_name} – Run {run_idx+1}/10")
            print(f"{'='*50}\n")
            run_main(model_name=model_name, run_idx= run_idx)