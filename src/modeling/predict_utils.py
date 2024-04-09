import numpy as np
import os, sys
import time
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from .models.seresnet18 import resnet18
from ..dataloader.dataset import ECGDataset, get_transforms
from .metrics import cal_multilabel_metrics, roc_curves
import pickle

class Predicting(object):
    def __init__(self, args):
        self.args = args
    
    def setup(self):
        ''' Initializing the device conditions and dataloader,
        loading trained model
        '''
        # Consider the GPU or CPU condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = self.args.device_count
            self.args.logger.info('using {} gpu(s)'.format(self.device_count))
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            self.args.logger.info('using {} cpu'.format(self.device_count))
        
        # Find test files based on the test csv (for naming saved predictions)
        # The paths for these files are in the 'path' column
        filenames = pd.read_csv(self.args.test_path, usecols=['path']).values.tolist()
        self.filenames = [f for file in filenames for f in file]

        # Load the test data
        testing_set = ECGDataset(self.args.test_path, 
                                 get_transforms('test'))
        channels = testing_set.channels
        self.test_dl = DataLoader(testing_set,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=(True if self.device == 'cuda' else False),
                                  drop_last=True)
        
        # Load the trained model
        self.model = resnet18(in_channel=channels,
                         out_channel=len(self.args.labels))
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.device))

        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)
        self.model.to(self.device)
        
    def predict(self):
        ''' Make predictions
        '''
        self.args.logger.info('predict() called: model={}, device={}'.format(
              type(self.model).__name__,
              self.device))

        # Saving the history
        history = {}
        history['test_micro_avg_prec'] = 0.0
        history['test_micro_auroc'] = 0.0
        history['test_macro_avg_prec'] = 0.0
        history['test_macro_auroc'] = 0.0
        history['test_challenge_metric'] = 0.0
        
        history['labels'] = self.args.labels
        history['test_csv'] = self.args.test_path
        history['threshold'] = self.args.threshold
        
        start_time_sec = time.time()
 
        # --- EVALUATE ON TESTING SET ------------------------------------- 
        self.model.eval()
        labels_all = torch.tensor((), device=self.device)
        logits_prob_all = torch.tensor((), device=self.device)  
        
        for i, (ecgs, ag, labels) in enumerate(self.test_dl):
            ecgs = ecgs.to(self.device) # ECGs
            ag = ag.to(self.device) # age and gender
            labels = labels.to(self.device) # diagnoses in SMONED CT codes 

            with torch.set_grad_enabled(False):  
                
                logits = self.model(ecgs, ag)
                logits_prob = self.sigmoid(logits)
                labels_all = torch.cat((labels_all, labels), 0)
                logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

            if i % 1000 == 0:
                self.args.logger.info('{:<4}/{:>4} predictions made'.format(i+1, len(self.test_dl)))

        # Predicting metrics
        test_macro_avg_prec, test_micro_avg_prec, test_macro_auroc, test_micro_auroc, test_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
        
        self.args.logger.info('macro avg prec: {:<6.2f} micro avg prec: {:<6.2f} macro auroc: {:<6.2f} micro auroc: {:<6.2f} challenge metric: {:<6.2f}'.format(
            test_macro_avg_prec,
            test_micro_avg_prec,
            test_macro_auroc,
            test_micro_auroc,
            test_challenge_metric))
        
        # Draw ROC curve for predictions
        roc_curves(labels_all, logits_prob_all, self.args.labels, epoch=None, save_path=self.args.roc_save_dir)
        
        # Add information to testing history
        history['test_micro_auroc'] = test_micro_auroc
        history['test_micro_avg_prec'] = test_micro_avg_prec
        history['test_macro_auroc'] = test_macro_auroc
        history['test_macro_avg_prec'] = test_macro_avg_prec
        history['test_challenge_metric'] = test_challenge_metric
        
        # Save the history
        history_savepath = os.path.join(self.args.output_dir,
                                        self.args.yaml_file_name + '_test_history.pickle')
        with open(history_savepath, mode='wb') as file:
            pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)

        # Save the labels and logits
        filenames = [os.path.basename(file) for file in self.filenames]
        logits_csv_path = os.path.join(self.args.output_dir,
                                        self.args.yaml_file_name + '_test_logits.csv') 
        labels_csv_path = os.path.join(self.args.output_dir,
                                        self.args.yaml_file_name + '_test_labels.csv') 

        logits_numpy = logits_prob_all.cpu().detach().numpy().astype(np.float32)
        logits_df = pd.DataFrame(logits_numpy, columns=self.args.labels, index=filenames)
        logits_df.to_csv(logits_csv_path, sep=',')
        
        labels_numpy = labels_all.cpu().detach().numpy().astype(np.float32)
        labels_df = pd.DataFrame(labels_numpy, columns=self.args.labels, index=filenames)
        labels_df.to_csv(labels_csv_path, sep=',')
            
        torch.cuda.empty_cache()
        
        end_time_sec = time.time()
        total_time_sec = end_time_sec - start_time_sec
        self.args.logger.info('Time total:     %5.2f sec' % (total_time_sec))