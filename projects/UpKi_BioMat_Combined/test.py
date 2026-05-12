import time
import numpy as np
import torch
import wandb
from modelbuilder import ModelBuilder



class Test:
    def run_testing(self, config, model, test_dataloader):
        self.config = config
        self.device = config['device']
        self.loss = self.config['loss']
        self.weight = self.config['loss_weight']
        self.model_name = self.config['model_name']
        self.classification = config['classification']
        self.n_output = len(self.config['selected_opensim_labels'])#Liste der Gelenkwinkel. Anzahl (3)
        if not self.n_output == len(self.weight):#prüft ob es eine Gewichtung der Gelenke gibt
            self.weight = None
        modelbuilder_handler = ModelBuilder(self.config)
        criterion = modelbuilder_handler.get_criterion(self.weight)#erstellt nur den Loss neu, nicht das ganze Modell
        self.tester = self.setup_tester()#wählt Testmodel aus
        y_pred, y_true, loss = self.tester(model, test_dataloader, criterion, self.device)#führt testing fkt. aus
        return y_pred, y_true, loss

    def setup_tester(self):
        if (self.model_name == 'transformer' and not self.classification) or (self.model_name == 'transformertsai' and not self.classification):
            tester = self.testing_transformer
        elif self.classification:
            tester = self.testing_w_classification
        else:
            tester = self.testing
        return tester

    def testing(self, model, test_dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_trues = []
            inference_times = [] #misst wie lange prediction dauert
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                start_time = time.time()
                #Forward Pass. Daten durch NN schicken
                y_pred = model(x.float())
                inference_times.append(time.time() - start_time)
                loss = criterion(y, y_pred) #loss berechnen
                test_loss.append(loss.item())
                test_preds.append(y_pred)
                test_trues.append(y)
            test_loss = torch.mean(torch.tensor(test_loss)) #Zusammenfassen zu Durchschnittswert
            print('Test Accuracy of the model: {}'.format(test_loss))
        # wandb.log({"inference time": np.mean(np.array(inference_times))})
        # wandb.log({"Test Loss": test_loss})
        return torch.cat(test_preds, 0), torch.cat(test_trues, 0), test_loss

    def testing_transformer(self, model, test_dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_trues = []
            inference_times = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_time = time.time()
                y_pred = model(x.float())  # just for transformer
                inference_times.append(time.time()-start_time)
                loss = criterion(y, y_pred.to(device))
                test_loss.append(loss.item())
                test_preds.append(y_pred)
                test_trues.append(y)
            test_loss = torch.mean(torch.tensor(test_loss))
            print('Test Accuracy of the model: {}'.format(test_loss))
        # wandb.log({"Test Loss": test_loss})
        # wandb.log({"inference time": np.mean(np.array(inference_times))})
        return torch.cat(test_preds, 0), torch.cat(test_trues, 0), test_loss
