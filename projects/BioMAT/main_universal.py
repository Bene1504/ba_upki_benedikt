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

#starte neuen wandb Run
wandb.init(project='CAMARGO_Dataset_Transformer')


def run_main():
    #Zufallszahlen-Generatoren
    torch.manual_seed(0)
    np.random.seed(42)
    dataset_name = 'camargo'
    #Dictionary mit allen Einstellungen(Hyperparameter, Pfade, Modelleinstellungen)
    config = get_config_universal(dataset_name)
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
                                                                                     kihadataset_test['y'], kihadataset_test['labels'])
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
            train_dataset = DataSetBuilder(kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'],
                                           transform_method=config['data_transformer'], scaler=None)
            #Reihenfolge in der Batches fürs Trainieren geladen werden solle
            #zufällig damit model sich nicht auf spezifische Reihenfolge optimiert
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
            test_dataset = DataSetBuilder(kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'],
                                          transform_method=config['data_transformer'], scaler=train_dataset.scaler)
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
            # Evaluation
            if individual_plot:
                for subject in config['test_subjects']:
                    subject_index = kihadataset_test['labels'][kihadataset_test['labels']['subject'] == subject].index.values
                    wandb_plotly_true_pred(y_true[subject_index], y_pred[subject_index], config['selected_opensim_labels'], str('test_'+ subject))
                    Evaluation(config=config, y_pred=y_pred[subject_index], y_true=y_true[subject_index], val_or_test=subject)
        # Evaluation
        for activity in config['test_activity']:
            activity_to_evaluate = activity
            activity_index = get_activity_index_test(kihadataset_test['labels'], activity_to_evaluate)
            wandb_plotly_true_pred(y_true[activity_index], y_pred[activity_index], config['selected_opensim_labels'], 'test_' + activity_to_evaluate)
            Evaluation(config=config, y_pred=y_pred[activity_index], y_true=y_true[activity_index], val_or_test='all_' + activity_to_evaluate)
            

if __name__ == '__main__':
    run_main()