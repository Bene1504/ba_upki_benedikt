import torch
import numpy as np
import wandb
from scipy.stats import pearsonr


class Evaluation:
    def __init__(self, config, y_pred, y_true, val_or_test, epoch=None):
        self.epoch = epoch
        self.val_or_test = val_or_test
        self.config = config
        self.selected_opensim_labels = self.config['selected_opensim_labels']
        if torch.is_tensor(y_pred):
            self.y_pred = y_pred.detach().cpu().numpy()
        else:
            self.y_pred = y_pred
        if torch.is_tensor(y_true):
            self.y_true = y_true.detach().cpu().numpy()
        else:
            self.y_true = y_true

        self.rmse_all = None
        self.run_evaluation()

    def run_evaluation(self):
        self.mae_all    = self.mae()
        self.rmse_all   = self.rmse()
        self.n_rmse_all = self.n_rmse()
        self.r_all      = self.correlation()
        mae ={}
        rmse = {}
        n_rmse ={}
        r = {}
        #for i, output in enumerate(self.selected_opensim_labels):
           # mae[self.val_or_test + '_MAE_' + output] = mae_all[i]
            #rmse[self.val_or_test+'_RMSE_'+ output] = rmse_all[i]
            #n_rmse[self.val_or_test+'_nRMSE_' + output] = n_rmse_all[i]
            #r[self.val_or_test+'_r_'+ output] = r_all[i]

        #mae[self.val_or_test+'_MAE_mean'] = np.array(mae_all).mean()
        #rmse[self.val_or_test+'_RMSE_mean'] = np.array(rmse_all).mean()
        #n_rmse[self.val_or_test+'_nRMSE_mean'] = np.array(n_rmse_all).mean()
        #r[self.val_or_test+'_r_mean'] = np.array(r_all).mean()

        #mae[self.val_or_test + '_MAE_std'] = np.array(mae_all).std()
        #rmse[self.val_or_test+'_RMSE_std'] = np.array(rmse_all).std()
        #n_rmse[self.val_or_test+'_nRMSE_std'] = np.array(n_rmse_all).std()
        #r[self.val_or_test+'_r_std'] = np.array(r_all).std()
        log_dict = {}
        if self.config['training_style'] != 'lopo':
            for i, output in enumerate(self.selected_opensim_labels):
                log_dict[self.val_or_test + '_MAE_'   + output] = self.mae_all[i]
                log_dict[self.val_or_test + '_RMSE_'  + output] = self.rmse_all[i]
                log_dict[self.val_or_test + '_nRMSE_' + output] = self.n_rmse_all[i]
                log_dict[self.val_or_test + '_r_'     + output] = self.r_all[i]

        log_dict[self.val_or_test + '_MAE_mean']   = np.array(self.mae_all).mean()
        log_dict[self.val_or_test + '_RMSE_mean']  = np.array(self.rmse_all).mean()
        log_dict[self.val_or_test + '_nRMSE_mean'] = np.array(self.n_rmse_all).mean()
        log_dict[self.val_or_test + '_r_mean']     = np.array(self.r_all).mean()

        log_dict[self.val_or_test + '_MAE_std']   = np.array(self.mae_all).std()
        log_dict[self.val_or_test + '_RMSE_std']  = np.array(self.rmse_all).std()
        log_dict[self.val_or_test + '_nRMSE_std'] = np.array(self.n_rmse_all).std()
        log_dict[self.val_or_test + '_r_std']     = np.array(self.r_all).std()


        if self.config['training_style'] != 'lopo':
            wandb.run.summary.update(log_dict)
            wandb.log(log_dict)

        print('RMSE Mean', self.val_or_test, ':', log_dict[self.val_or_test + '_RMSE_mean'])
        #print('RMSE Mean', self.val_or_test,': ', rmse[self.val_or_test+'_RMSE_mean'])
        #print('RMSE:', rmse)
        

    def mae(self):
        mae_all = []
        for i in range(self.y_pred.shape[2]):
            ae = abs(self.y_pred[:, :, i] - self.y_true[:, :, i])
            ame = ae.mean(axis=1).mean(axis=0)
            mae_all.append(ame)
        return mae_all


    def rmse(self):
        rmse_all = []
        for i in range(self.y_pred.shape[2]):
            se = (self.y_pred[:, :, i] - self.y_true[:, :, i])**2
            mse = se.mean(axis=1).mean(axis=0)
            rmse = np.sqrt(mse)
            rmse_all.append(rmse)
        return rmse_all

    def n_rmse(self):
        n_rmse_all = []
        for i in range(self.y_pred.shape[2]):
            se = (self.y_pred[:, :, i] - self.y_true[:, :, i])**2
            mse = se.mean(axis=1).mean(axis=0)
            rmse = np.sqrt(mse)
            nrmse = 100 * rmse / (self.y_true[:, :, i].max() - self.y_true[:, :, i].min()).item()
            n_rmse_all.append(nrmse)
        return n_rmse_all

    def correlation(self):
        r_all = []
        for i in range(self.y_pred.shape[2]):
            r = np.array([pearsonr(self.y_pred[j, :, i], self.y_true[j, :, i])[0] for j in range(len(self.y_true[:, :, i]))]).mean()
            r_all.append(r)
        return r_all


    # for i in test_subject:
        # build train dataset
        # build test dataset
        # run parameters tunning
        # run train
        # run test
        # save loss for this test subject


    def load_trained_model(self):
        pass


    def test(self, model, device, test_dataloader, n_output, criterion):
        model.eval()
        with torch.no_grad():
            test_loss = []
            outputs = []
            labels = []
            # for input, label_cont, label_disc in test_dataloader:
            for input, label_cont in test_dataloader:
                input = input.to(device)
                label = label_cont.to(device)
                output = model(input, n_output)
                output = torch.stack(output, dim=0)
                output = output.permute(1, 2, 0)
                test_loss.append(criterion(output, label).item())
                outputs.append(output)
                labels.append(label)
            test_loss = torch.mean(torch.tensor(test_loss))
            print('Test Loss of the model: {}'.format(test_loss))
        wandb.log({"Test Loss": test_loss.item()})
        return torch.cat(outputs, 0), torch.cat(labels, 0)
    
    def setup_wandb_metrics(opensim_labels):
        """
        wird einmalig in main aufgerufen

        registriert alle metriken mit 'epoch' als x-Achse
        gegen step warnings
        """
        wandb.define_metric("epoch")

        # Trainings-Kurven
        for name in ["Train Loss", "Validation Loss", "training time", "n parameters"]:
            wandb.define_metric(name, step_metric="epoch")

        # Val-Metriken pro Gelenk und Aggregat
        for label in opensim_labels:
            for metric in ["RMSE", "MAE", "nRMSE", "r"]:
                wandb.define_metric(f"val_{metric}_{label}", step_metric="epoch")

        for suffix in ["mean", "std"]:
            for metric in ["RMSE", "MAE", "nRMSE", "r"]:
                wandb.define_metric(f"val_{metric}_{suffix}", step_metric="epoch")

    def log_epoch(config, y_pred, y_true, epoch, train_loss, val_loss=None):
        """
        pro epoche aus train.py aufgerufen:
        """
        labels = config['selected_opensim_labels']

        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()
        if torch.is_tensor(y_true):
            y_true = y_true.detach().cpu().numpy()

        log_dict = {
            "Train Loss": float(train_loss),
            "epoch":      epoch,
        }
        if val_loss is not None:
            log_dict["Validation Loss"] = float(val_loss)

        rmse_all, mae_all = [], []

        for i, label in enumerate(labels):
            se   = (y_pred[:, :, i] - y_true[:, :, i]) ** 2
            rmse = float(np.sqrt(se.mean()))
            mae  = float(np.abs(y_pred[:, :, i] - y_true[:, :, i]).mean())

            log_dict[f"val_RMSE_{label}"] = rmse
            log_dict[f"val_MAE_{label}"]  = mae
            rmse_all.append(rmse)
            mae_all.append(mae)

        log_dict["val_RMSE_mean"] = float(np.mean(rmse_all))
        log_dict["val_RMSE_std"]  = float(np.std(rmse_all))
        log_dict["val_MAE_mean"]  = float(np.mean(mae_all))
        log_dict["val_MAE_std"]   = float(np.std(mae_all))

        wandb.log(log_dict, step=epoch)