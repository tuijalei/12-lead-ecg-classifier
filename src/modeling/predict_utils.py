import numpy as np
import os, sys
import torch
from torch import nn
import warnings
import math
import pandas as pd
from scipy.io import loadmat
from torch.utils.data import DataLoader

from .models.seresnet18 import resnet18
from ..dataloader.dataset import ECGDataset, get_transforms
from .metrics import cal_multilabel_metrics

class Predicting(object):
    def __init__(self, args):
        self.args = args
        
        # Consider the GPU or CPU condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = self.args.device_count
            print('using {} gpu(s)'.format(self.device_count))
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            print('using {} cpu'.format(self.device_count))
    
    def predict(self):
        ''' Make predictions
        '''

        # Find test files based on the test csv
        # The paths for these files are in the 'path' column
        file_names = pd.read_csv(self.args.test_path, usecols=['path']).values.tolist()
        file_names = [f for file in file_names for f in file]

        # Load the test data
        testing_set = ECGDataset(self.args.test_path, 
                                 get_transforms('test', self.args.normalizetype, self.args.seq_length))
        channels = testing_set.channels
        test_loader = DataLoader(testing_set,
                                     batch_size=1,
                                     shuffle=False)
    
        # Load the trained model
        self.load_model(channels)
        sigmoid = nn.Sigmoid()
        
        # Iterate over test data
        for i, (ecg, ag, labels) in enumerate(test_loader):
            ecg = ecg.to(self.device)
            ag = ag.to(self.device)

            with torch.no_grad():
                logits = self.model(ecg, ag)
                logits_prob = sigmoid(logits)

            score, pred_label = self.output_label(logits_prob, len(self.args.labels))
            pred_label = np.squeeze(pred_label.astype(np.int))
            score = np.squeeze(score)
            
            if i % 200 == 0:
                print('{}/{} predictions made'.format(i+1, len(test_loader)))

            # Save the prediction
            self.save_predictions(file_names[i], score, pred_label)


    def load_model(self, channels):
        ''' Load a trained model from disk and set it to evaluation mode

        :param channels: Number of input channels used 
        :type model_path: int
        '''

        self.model = resnet18(in_channel= channels,
                         out_channel=len(self.args.labels))

        # Load the model based on the device condition
        if torch.cuda.is_available():
            
            if self.args.device_count > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model.module.load_state_dict(torch.load(self.args.model_path))
                
        else:
            self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

        
    def output_label(self, logits_prob, num_classes, threshold=0.5):
        ''' Get output labels and probability scores 

        :param logits_prob: The probabilities of each class belonging to a disease class
        :type logits_prob: numpy.ndarray
        :param num_classes: The number of classes used in the classification task
        :type num_classes: int
        :param threshold: Decision threshold
        :type: float

        :return score_temp: Probabilities for the predicted labels
        :return pred_label: Predicted label in binary form
        :rtypes: numpy.ndarray
        '''

        pred_label = np.zeros(num_classes, dtype=int)

        # Most likely class is the one which probability is the highest
        _, y_pre_label = torch.max(logits_prob, 1)
        y_pre_label = y_pre_label.cpu().detach().numpy()
        pred_label[y_pre_label] = 1

        # Threshold check
        scores = logits_prob.cpu().detach().numpy()
        y_pre = (scores - threshold) >= 0
        pred_label = pred_label + y_pre
        pred_label[pred_label > 1.1] = 1

        return scores, pred_label


    def save_predictions(self, filename, scores, labels):
        '''Save the challenge predictions in csv file with record id, 
        diagnoses predicted and their confidence score as

        #Record ID
        164889003, 270492004, 164909002, 426783006, 59118001, 284470004,  164884008,
        1,         1,         0,         0,         0,        0,          0,        
        0.9,       0.6,       0.2,       0.05,      0.2,      0.35,       0.35,     
        '''

        recording = os.path.basename(os.path.splitext(filename)[0])
        new_file = os.path.basename(filename.replace('.mat','.csv'))
        output_file = os.path.join(self.args.output_dir, new_file)

        # Include the filename as the recording number
        recording_string = '#{}'.format(recording)
        class_string = ','.join(self.args.labels)
        label_string = ','.join(str(i) for i in labels)
        score_string = ','.join(str(i) for i in scores)

        # Write the output file
        with open(output_file, 'w') as file:
            file.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')
           