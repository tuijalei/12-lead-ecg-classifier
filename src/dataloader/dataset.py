import torch
from torch.utils.data import Dataset
import pandas as pd
from .dataset_utils import load_data, encode_metadata
from .transforms import *


def get_transforms(dataset_type, aug_type=None, precision=None):
    ''' Get transforms for ECG data based on the dataset type (train, validation, test)
    '''
    seq_length = 4096
    normalizetype = '0-1' 
    
    preprocess = {
        
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
    
    if dataset_type == 'train':

        # Add augmentations only to training data if augmentation type set
        if aug_type is not None:

            if aug_type == 'round' and precision is None:
                raise Exception('If the Round class used, the precision needs to be set!')

            transforms = {

                'round': Compose([
                    Round(precision)
                    ], p = 1.0),

                'noise': Compose([
                    AddNoise(p = 0.5)
                    ], p = 1.0),

                'roll': Compose([
                    Roll(p = 0.5)
                    ], p = 1.0),

                'flip_x': Compose([
                    Flipx(p = 0.5)
                    ], p = 1.0),

                'flip_y': Compose([
                    Flipy(p = 0.5)
                    ], p = 1.0),

                'multiply_sine': Compose([
                    MultiplySine(p = 0.5)
                    ], p = 1.0),

                'multiply_linear': Compose([
                    MultiplyLinear(p = 0.5)
                    ], p = 1.0),

                'multiply_triangle': Compose([
                    MultiplyTriangle(p = 0.5)
                    ], p = 1.0),

                'rand_stretch': Compose([
                    RandomStretch(p = 0.5)
                    ], p = 1.0),

                'resample_linear': Compose([
                    ResampleLinear(p = 0.5)
                    ], p = 1.0),

                'notch': Compose([
                    NotchFilter(fs=250, p = 0.5)
                    ], p = 1.0),
            }

        return preprocess[dataset_type], transforms[aug_type]
    else:
        return preprocess[dataset_type], None


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

        self.preprocess, self.transforms = transforms
        self.channels = 12
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        file_name = self.data[item]
        fs = self.fs[item]
        ecg = load_data(file_name)


        # 1) Preprocess
        ecg = self.preprocess(ecg)
        print(ecg)
        # 2) Add augmentations
        if self.transforms is not None:
            ecg = self.transforms(ecg)
            print('Transforms added! (', self.transforms.transforms, ')')
            
        label = self.multi_labels[item]
        
        age = self.age[item]
        gender = self.gender[item]
        age_gender = encode_metadata(age, gender)
        print(ecg)
        print('------')
        return ecg, torch.from_numpy(age_gender).float(), torch.from_numpy(label).float()
      