import numpy as np, os
from torch.utils.data import Dataset
import pandas as pd
from .dataset_utils import load_data, encode_metadata, normalize, apply_filter, standardize_sampling_rate


class ECGWindowPaddingDataset(Dataset):
    def __init__(self, csv_path, window, nb_windows, filter_bandwidth, all_features):
        ''' Return randome window length segments from ecg signal, pad if window is too large
            df: trn_df, val_df or tst_df
            window: ecg window length e.g 2500 (5 seconds)
            nb_windows: number of windows to sample from record
        '''

        # Load metadata from CSV
        df = pd.read_csv(csv_path)
        self.data = df['path'].tolist()
        labels = df.iloc[:, 4:].values
        self.multi_labels = [labels[i, :] for i in range(labels.shape[0])]
        self.fs = df['fs'].tolist()

        self.window = window
        self.nb_windows = nb_windows
        self.filter_bandwidth = filter_bandwidth
        self.all_features = all_features

        # Compute mean and std for TOP features
        feats = all_features.iloc[:, 1:].values # Filenames in the first index
        feats[np.isinf(feats)] = np.nan
        self.feats_mean = np.nanmean(feats, axis=0)
        self.feats_std =  np.nanstd(feats, axis=0)

        # Convert age==-1 (unknown) to zero and encode the gender values
        ages = np.array([age if age > 0 else 0 for age in df['age'].tolist()])[None].T
        self.age = (ages - np.nanmean(ages)) / np.nanstd(ages)
        self.gender = np.array([1. if h.find('Female') >= 0. else 0 for h in df['gender'].tolist()])[None].T

        self.channels = 12

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        
        # Get ECG data
        file_name = self.data[item]
        ecg = load_data(file_name)
        ecg_fs = self.fs[item]

        # Standardize sampling rate
        ecg = standardize_sampling_rate(ecg, ecg_fs)

        # Length of the ECG signal
        seq_len = ecg.shape[-1]

        # Get top features based on the filename
        top_feats = self.all_features[self.all_features.file_name == os.path.basename(file_name)].iloc[:, 1:].values
        # First, convert any infs to nans
        top_feats[np.isinf(top_feats)] = np.nan
        # Replace NaNs with feature means
        top_feats[np.isnan(top_feats)] = self.feats_mean[None][np.isnan(top_feats)]
        # Normalize wide features
        feats_normalized = (top_feats - self.feats_mean) / self.feats_std
        # Use zeros (normalized mean) if cannot find patient features
        if not len(feats_normalized):
            feats_normalized = np.zeros(len(self.feats_mean))[None]

        # Apply band pass filter
        if self.filter_bandwidth is not None:
            ecg = apply_filter(ecg, self.filter_bandwidth)

        # Normalize ECG sequences between -1 and 1    
        ecg = normalize(ecg)

        # Gather metadata
        label = self.multi_labels[item]
        age = self.age[item]
        gender = self.gender[item]

        # Add just enough padding to allow window
        pad = np.abs(np.min(seq_len - self.window, 0))
        if pad > 0:
            ecg = np.pad(ecg, ((0,0),(0,pad+1)))
            seq_len = ecg.shape[-1] # get the new length of the ecg sequence
        
        starts = np.random.randint(seq_len - self.window + 1, size=self.nb_windows) # get start indices of ecg segment     
        ecg_segs = np.array([ecg[:,start:start+self.window] for start in starts])
        
        # Gather all metadata into one numpy.ndarray ([age, gender, <normalized top features>])
        feats_normalized = np.concatenate((age, gender, feats_normalized.squeeze()))
        return ecg_segs, feats_normalized, label