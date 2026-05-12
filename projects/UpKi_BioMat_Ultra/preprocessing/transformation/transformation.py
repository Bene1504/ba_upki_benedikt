from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler,QuantileTransformer
from scipy.signal import butter, filtfilt
import numpy as np
import copy
from .hierarchical_scaler import HierarchicalScaler

class Transformation:
    def __init__(self, method, by, train_sids=None, test_sid=None):
        self.method = method
        self.by = by
        self.train = []
        self.val = []
        self.test = []
        self.train_sids = train_sids
        self.test_sid  = test_sid

    def get_scaler(self):
        if self.method == 'StandardScaler':
            scaler = StandardScaler()
        elif self.method == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif self.method == 'HierarchicalScaler':      # ← neu
            scaler = HierarchicalScaler()
        elif self.method == 'RobustScaler':      # ← neu
            scaler = RobustScaler()
        elif self.method == 'QuantileTransformer':      # ← neu
            scaler = QuantileTransformer()
        return scaler

    def get_by(self, setting):
        if self.by == 'by_sample':
            by_axis = 0
        elif self.by == 'by_var':
            by_axis = 2
        elif self.by == 'by_step':
            by_axis = 1
        elif self.by == 'by_groupvar':
            if setting == "train":
                n_channels = self.train.shape[2]          # z.B. 24
            elif setting == "test":
                n_channels = self.test.shape[2] 
                
            n_features = 6  # 6 (3 acc + 3 gyro)
            n_sensors  = n_channels // n_features     # z.B. 4
    
            acc_idx = []
            gyr_idx = []
            for s in range(n_sensors):
                base = s * n_features                 # 0, 6, 12, 18
                acc_idx += list(range(base, base + 3))      # 0,1,2 | 6,7,8 | ...
                gyr_idx += list(range(base + 3, base + 6))  # 3,4,5 | 9,10,11 | ...
    
            by_axis = {'acc': acc_idx, 'gyr': gyr_idx}
        return by_axis

    def transform_train(self):
        scaler = self.get_scaler()# Standardscaler
        by_axis = self.get_by("train")
        
        if self.method == 'HierarchicalScaler':
            n_samples, timesteps, n_joints = self.train.shape
            
            # Fit mit originaler Shape — mask stimmt dann
            scaler.fit(self.train, self.train_sids)  # ← 3D Array übergeben
            self.scaler_fit = scaler
        
            train_scaled = np.zeros_like(self.train)
            for i, sid in enumerate(self.train_sids):
                train_scaled[i] = scaler.transform(self.train[i], subject_id=sid)
            self.train = train_scaled
            return  
     
        
        
        if self.by == 'by_groupvar':
            acc = self.train[:, :, by_axis['acc']]
            acc_scaler_fit = scaler.fit(acc.reshape(-1,1))

            train_old=copy.deepcopy(self.train)
            
            self.train[:, :, by_axis['acc']] = acc_scaler_fit.transform(acc.reshape(-1,1)).reshape(
                acc.shape)
            gyr = self.train[:, :, by_axis['gyr']]
            gyr_scaler_fit = scaler.fit(gyr.reshape(-1,1))
            self.train[:, :, by_axis['gyr']] = gyr_scaler_fit.transform(gyr.reshape(-1,1)).reshape(gyr.shape)
            self.scaler_fit = {'acc_scaler_fit':acc_scaler_fit, 'gyr_scaler_fit':gyr_scaler_fit}
            
            train_new=self.train
        else:
            self.scaler_fit = scaler.fit(self.train.reshape(-1, self.train.shape[by_axis]))

            train_old=copy.deepcopy(self.train)
            self.train = self.scaler_fit.transform(self.train.reshape(-1, self.train.shape[by_axis])).reshape(self.train.shape)
            train_new=self.train
            
    def transform_val(self):
        by_axis = self.get_by('test')
        if self.by == 'by_groupvar':
            acc = self.val[:, :, by_axis['acc']]
            self.val[:, :, by_axis['acc']] = self.scaler_fit['acc_scaler_fit'].transform(acc.reshape(-1,1)).reshape(acc.shape)
            gyr = self.val[:, :, by_axis['gyr']]
            self.val[:, :, by_axis['gyr']] = self.scaler_fit['gyr_scaler_fit'].transform(gyr.reshape(-1, 1)).reshape(gyr.shape)
        else:
            self.val = self.scaler_fit.transform(self.val.reshape(-1, self.val.shape[by_axis])).reshape(self.val.shape)

    def transform_test(self):
        
        # ── Hierarchical ───────────────────────────────────────────────────────
        if self.method == 'HierarchicalScaler':
            # Test-Proband: Median als Subject-Mean schätzen
            # self.test_sid: string — der eine Test-Proband
            self.scaler_fit.subject_mean[self.test_sid] = np.median(
                self.test.reshape(-1, self.test.shape[2]), axis=0
            )
            test_scaled = np.zeros_like(self.test)
            for i in range(len(self.test)):
                test_scaled[i] = self.scaler_fit.transform(
                    self.test[i], subject_id=self.test_sid
                )
            self.test = test_scaled
            return
        
        by_axis = self.get_by("test")
        if self.by == 'by_groupvar':
            acc = self.test[:, :, by_axis['acc']]
            self.test[:, :, by_axis['acc']] = self.scaler_fit['acc_scaler_fit'].transform(acc.reshape(-1,1)).reshape(acc.shape)
            gyr = self.test[:, :, by_axis['gyr']]
            self.test[:, :, by_axis['gyr']] = self.scaler_fit['gyr_scaler_fit'].transform(gyr.reshape(-1, 1)).reshape(gyr.shape)
        else:
            self.test = self.scaler_fit.transform(self.test.reshape(-1, self.test.shape[by_axis])).reshape(self.test.shape)

    def run_transform(self, train=None, val=None, test=None, scaler_fit=None, mode='2d'):
        self.train = train
        self.val = val
        self.test = test
        self.scaler_fit = scaler_fit
        
        if train is not None and scaler_fit is None:# Trainingsteil
            if mode == '3d':
                self.transform_train_3d()
            else:
                self.transform_train()
            return self.scaler_fit, self.train
        
        if train is None and scaler_fit is not None:
            if val is not None and test is not None:
                if mode == '3d':
                    self.transform_val_3d()
                    
                else:
                    self.transform_val()
                    self.transform_test()
                return self.val, self.test
            

            elif val is not None and test is None:
                if mode == '3d':
                    self.transform_val_3d()
                else:
                    self.transform_val()
                return self.val

            elif val is None and test is not None:
                if mode == '3d':
                    self.transform_test_3d() 
                else:
                    self.transform_test()
                return self.test

                # if val is not None and test is not None:
        #     self.transform_val()
        #     self.transform_test()
        #     return self.scaler_fit, self.train, self.val, self.test
        #
        # elif val is not None and test is None:
        #     self.transform_val()
        #     return self.scaler_fit, self.train, self.val
        #
        # elif val is None and test is None:
        #     return self.scaler_fit, self.train




    def transform_train_3d(self):
        """Für 3D Arrays (N, timesteps, features)"""
        # Optional: Lowpass Filter vor dem Scaling
        import numpy as np
        import copy

            
        orig_shape = self.train.shape  # (N, timesteps, 7)
        
        scaler = self.get_scaler()
        
        if self.method == 'HierarchicalScaler':
            self.scaler_fit = scaler.fit(self.train, self.train_sids)  # ← 3D + sids
            
            train_scaled = np.zeros_like(self.train)
            for i, sid in enumerate(self.train_sids):
                train_scaled[i] = scaler.transform(self.train[i], subject_id=sid)
            self.train = train_scaled
            return
    
        # ── Standard / MinMax / Robust / Quantile ─────────────────────────────
        flat = self.train.reshape(-1, orig_shape[2])  # (N*timesteps, 7)
        self.scaler_fit = scaler.fit(flat)
        self.train = self.scaler_fit.transform(flat).reshape(orig_shape)
        
    def transform_val_3d(self):
        """Wendet gefitteten Scaler auf Validierungs Daten an"""
        orig_shape = self.val.shape
        flat = self.val.reshape(-1, orig_shape[2])
        self.val = self.scaler_fit.transform(flat).reshape(orig_shape)
        
    def transform_test_3d(self):
        orig_shape = self.test.shape
    
        if self.method == 'HierarchicalScaler':
            # Test-Proband Offset aus Median schätzen
            self.scaler_fit.subject_mean[self.test_sid] = np.median(
                self.test.reshape(-1, orig_shape[2]), axis=0
            )
            test_scaled = np.zeros_like(self.test)
            for i in range(len(self.test)):
                test_scaled[i] = self.scaler_fit.transform(
                    self.test[i], subject_id=self.test_sid
                )
            self.test = test_scaled
            return
    
        # ── Standard / MinMax / Robust / Quantile ─────────────────────────────
        flat = self.test.reshape(-1, orig_shape[2])
        self.test = self.scaler_fit.transform(flat).reshape(orig_shape)  