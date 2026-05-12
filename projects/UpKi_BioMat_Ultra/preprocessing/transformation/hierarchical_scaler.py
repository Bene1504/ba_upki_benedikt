import numpy as np

class HierarchicalScaler:
    """
    Drop-in replacement für StandardScaler.
    Entfernt individuellen anatomischen Offset pro Proband
    und skaliert global über alle Trainingsprobanden.
    """
    def __init__(self):
        self.global_std   = None
        self.global_mean  = None
        self.subject_mean = {}

    def fit(self, y_train, subject_ids_train):
        """y_train: (n_samples, timesteps, n_joints) oder (n_samples, n_joints)"""
        subject_ids_train = np.array(subject_ids_train)
        
        # Flatten für globale Statistiken
        if y_train.ndim == 3:
            y_flat = y_train.reshape(-1, y_train.shape[2])  # (n_samples*timesteps, n_joints)
        else:
            y_flat = y_train
        
        self.global_std  = y_flat.std(axis=0) + 1e-8
        self.global_mean = y_flat.mean(axis=0)
    
        # Subject Mean: pro Sample mitteln, dann über Timesteps
        for sid in np.unique(subject_ids_train):
            mask = subject_ids_train == sid          # (n_samples,) ← stimmt jetzt
            subject_data = y_train[mask]             # (n_sid_samples, timesteps, n_joints)
            self.subject_mean[sid] = subject_data.reshape(-1, y_train.shape[-1]).mean(axis=0)
        
        return self

    def transform(self, y, subject_id):
        mean = self.subject_mean.get(subject_id, self.global_mean)
        return (y - mean) / self.global_std

    def inverse_transform(self, y, subject_id):
        mean = self.subject_mean.get(subject_id, self.global_mean)
        return (y * self.global_std) + mean