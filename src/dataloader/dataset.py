import torch
from torch.utils.data import Dataset
import pandas as pd
from .dataset_utils import load_data, encode_metadata
from .transforms import Compose, RandomClip, Normalize, ValClip, Retype


def get_transforms(dataset_type):
    ''' Get transforms for ECG data based on the dataset type (train, validation, test)
    '''
    seq_length = 4096
    normalizetype = '0-1'
    
    data_transforms = {
        
        'train': Compose([
            RandomClip(w=seq_length),
            Normalize(normalizetype),
            Retype() 
        ], p = 1.0),
        
        'val': Compose([
            ValClip(w=seq_length),
            Normalize(normalizetype),
            Retype()
        ], p = 1.0),
        
        'test': Compose([
            ValClip(w=seq_length),
            Normalize(normalizetype),
            Retype()
        ], p = 1.0)
    }
    return data_transforms[dataset_type]


class ECGDataset(Dataset):
    ''' Class implementation of Dataset of ECG recordings
    
    :param path: The directory of the data used
    :type path: str
    :param preprocess: Preprocess transforms for ECG recording
    :type preprocess: datasets.transforms.Compose
    :param transform: The other transforms used for ECG recording
    :type transform: datasets.transforms.Compose
    '''

    def __init__(self, path, transforms):
        df = pd.read_csv(path)
        self.data = df['path'].tolist()
        labels = df.iloc[:, 4:].values
        self.multi_labels = [labels[i, :] for i in range(labels.shape[0])]
        
        self.age = df['age'].tolist()
        self.gender = df['gender'].tolist()
        self.fs = df['fs'].tolist()

        self.transforms = transforms
        self.channels = 12
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        file_name = self.data[item]
        fs = self.fs[item]
        ecg = load_data(file_name)
        
        ecg = self.transforms(ecg)
        
        label = self.multi_labels[item]
        
        age = self.age[item]
        gender = self.gender[item]
        age_gender = encode_metadata(age, gender)
        return ecg, torch.from_numpy(age_gender).float(), torch.from_numpy(label).float()
      