import numpy as np
import os, sys
import torch
from torch import nn
import pickle
import time
import pandas as pd
from torch.utils.data import DataLoader
from .models.seresnet18 import resnet18
from ..dataloader.dataset import ECGDataset, get_transforms
from .metrics import cal_multilabel_metrics, roc_curves


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
            print('using {} gpu(s)'.format(self.device_count))
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            print('using {} cpu'.format(self.device_count))
        
        # Find test files based on the test csv (for naming saved predictions)
        # The paths for these files are in the 'path' column
        file_names = pd.read_csv(self.args.test_path, usecols=['path']).values.tolist()
        self.file_names = [f for file in file_names for f in file]

        # Load the test data
        testing_set = ECGDataset(self.args.test_path, 
                                 get_transforms('test', self.args.normalizetype, self.args.seq_length))
        channels = testing_set.channels
        self.test_dl = DataLoader(testing_set,
                                     batch_size=1,
                                     shuffle=False)
        
        # Load the trained model
        self.model = resnet18(in_channel=channels,
                         out_channel=len(self.args.labels))

        # Load the model based on the device condition
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)    
        else:
            self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.device))
        
        self.sigmoid = nn.Sigmoid()
        
        self.model.to(self.device)
        self.sigmoid.to(self.device)
        self.model.eval()
    
    def predict(self):
        ''' Make predictions
        '''

        # Saving the history
        history = {}
        history['test_micro_avg_prec'] = 0.0
        history['test_micro_auroc'] = 0.0
        history['test_macro_avg_prec'] = 0.0
        history['test_macro_auroc'] = 0.0
        
        history['labels'] = self.args.labels
        history['test_csv'] = self.args.test_path
        
        labels_all = torch.tensor((), device=self.device)
        logits_prob_all = torch.tensor((), device=self.device)  
        
        start_time_sec = time.time()
 
        # Iterate over test data
        for i, (ecg, ag, labels) in enumerate(self.test_dl):
            ecg = ecg.to(self.device) # ECGs
            ag = ag.to(self.device) # age and gender
            labels = labels.to(self.device) # diagnoses in SMONED CT codes 

            with torch.set_grad_enabled(False):
                
                logits = self.model(ecg, ag)
                logits_prob = self.sigmoid(logits)
                labels_all = torch.cat((labels_all, labels), 0)
                logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

            
            # Threshold check: if (prob-threshold) > 0, mark as 1
            pred_label = np.zeros(len(self.args.labels), dtype=int)
            scores = logits_prob.cpu().detach().numpy()
            y_pre = (scores - self.args.threshold) >= 0
            pred_label = pred_label + y_pre
            pred_label = np.squeeze(pred_label.astype(np.int))
            scores = np.squeeze(scores)
            
            # Save the prediction
            self.save_predictions(self.file_names[i], pred_label, scores)

            if i % 1000 == 0:
                print('{}/{} predictions made'.format(i+1, len(self.test_dl)))

        # Predicting metrics
        test_macro_avg_prec, test_micro_avg_prec, test_macro_auroc, test_micro_auroc = cal_multilabel_metrics(labels_all, logits_prob_all)
        
        print('Predicting metrics: macro avg prec: %5.2f, micro avg prec: %5.2f, macro auroc %5.2f, micro auroc: %5.2f' % \
                  (test_macro_avg_prec,
                   test_micro_avg_prec,
                   test_macro_auroc,
                   test_micro_auroc))
        
        roc_curves(labels_all, logits_prob_all, self.args.labels, save_path = self.args.output_dir)
        
        history['test_micro_auroc'] = test_micro_auroc
        history['test_micro_avg_prec'] = test_micro_avg_prec
        history['test_macro_auroc'] = test_macro_auroc
        history['test_macro_avg_prec'] = test_macro_avg_prec
        
        # Save the history
        history_savepath = os.path.join(self.args.output_dir, 'test_history.pickle')
        with open(history_savepath, mode='wb') as file:
            pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
            
        
        end_time_sec = time.time()
        total_time_sec = end_time_sec - start_time_sec
        print()
        print('Time total:     %5.2f sec' % (total_time_sec))

 
    def save_predictions(self, filename, labels, scores):
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
           