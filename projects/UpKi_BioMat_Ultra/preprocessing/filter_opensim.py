from scipy.signal import butter, filtfilt
from scipy.fft import fft
import numpy as np
from utils import filter



class FilterOpenSim:
    def __init__(self, y, lowcut=6, fs=100, order=2):
       self.y = y
       self.lowcut=lowcut
       self.fs = fs
       self.order = order

    def run_lowpass_filter(self):
        
        if isinstance(self.y, list):
            # Liste mit unterschiedlich langen Sequenzen
            filtered_y = []
            for seq in self.y:                          # pro Sequenz/Trial
                seq = np.array(seq)                     # (n_timesteps, n_joints)
                filtered_seq = np.zeros_like(seq)
                for j in range(seq.shape[1]):           # pro Joint
                    filtered_seq[:, j] = filter.butter_lowpass_filter(
                        seq[:, j], self.lowcut, self.fs, self.order
                    )
                filtered_y.append(filtered_seq)
            return filtered_y
    
        else:
            y = np.array(self.y)
            if y.ndim == 2:
                filtered_y = np.zeros_like(y)
                for j in range(y.shape[1]):
                    filtered_y[:, j] = filter.butter_lowpass_filter(
                        y[:, j], self.lowcut, self.fs, self.order
                    )
            elif y.ndim == 3:
                filtered_y = np.zeros_like(y)
                for w in range(y.shape[0]):
                    for j in range(y.shape[2]):
                        filtered_y[w, :, j] = filter.butter_lowpass_filter(
                            y[w, :, j], self.lowcut, self.fs, self.order
                        )
            return filtered_y