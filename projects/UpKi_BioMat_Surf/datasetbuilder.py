import numpy as np
import torch
from torch.utils.data import Dataset
from preprocessing.transformation.transformation import Transformation
import torch.nn.functional as F
from sklearn import preprocessing
import torch


class DataSetBuilder(Dataset):
    def __init__(self, x, y, labels, config, transform_method=None, scaler=None, y_scaler=None):
        self.config = config
        self.x = x
        self.y = y
        self.labels = labels
        self.y_label = []
        

        self.transform_method = transform_method
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.y_scaling = config['y_scaling']
        self._preprocess()
        self.n_sample = len(y)

        self.x = torch.from_numpy(self.x).double() 
        if isinstance(self.y, list): #zur Sicherheit, falls self.y eine Liste sein sollte
            self.y = np.array(self.y)
        self.y = torch.from_numpy(self.y).double()

    def _run_label_encoding(self):
        le = preprocessing.LabelEncoder()
        y_label = le.fit_transform(self.labels[:, 0, 3])
        y_label = torch.as_tensor(y_label)
        self.y_label = y_label.to(torch.int64)

    def _preprocess(self):
        if self.labels is not None:
            self.train_sids = self.labels[:, 0, 0]  # (n_samples,)
            
            all_subjects   = set(self.config['lopo_subjects'])
            train_subjects = set(np.unique(self.train_sids))
            missing        = all_subjects - train_subjects
            
            if len(missing) == 1:
                # Train Dataset → ein Subject fehlt = test_sid
                self.test_sid = missing.pop()
            elif len(train_subjects) == 1:
                # Test Dataset → nur ein Subject drin = das ist test_sid
                self.test_sid = train_subjects.pop()
            else:
                self.test_sid = None
            
            print(f"DEBUG test_sid: {self.test_sid}")  # ← zur Kontrolle
 
        if self.transform_method['data_transformer_method'] is not None:
            self._run_transform()

    def _run_transform(self):
        transform_handler = Transformation(
            method=self.transform_method['data_transformer_method'],
            by=self.transform_method['data_transformer_by'],train_sids=self.train_sids,
        )
        y_transform = Transformation(
            method=self.transform_method['data_transformer_method'],
            by='by_var',train_sids=self.train_sids,test_sid=self.test_sid  
        )

        if self.scaler is None:
            # Training:
            self.scaler, self.x = transform_handler.run_transform(
                train=self.x, scaler_fit=None
            )
            if self.y_scaling:
                self.y_scaler, self.y = y_transform.run_transform(train=self.y, scaler_fit=None, mode='3d')
            else:
                self.y_scaler = None
        else:
            # Test:
            self.x = transform_handler.run_transform(
                test=self.x, scaler_fit=self.scaler
            )
            if self.y_scaling:
                self.y = y_transform.run_transform(test=self.y, scaler_fit=self.y_scaler, mode='3d')
            
        '''# Val:
            self.x = transform_handler.run_transform(
                test=self.x, scaler_fit=self.scaler
            )
            if self.y_scaling:
                self.y = y_transform.run_transform(test=self.y, scaler_fit=self.y_scaler, mode='3d')'''


    def __len__(self):
        return self.n_sample

    def __getitem__(self, item):
        return self.x[item], self.y[item]