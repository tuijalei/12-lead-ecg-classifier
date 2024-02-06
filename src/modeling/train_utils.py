import os, sys
import time
import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .optimizer import NoamOpt
from .models.ctn import CTN
from ..dataloader.dataset import ECGWindowPaddingDataset
from .metrics import cal_multilabel_metrics, roc_curves, physionet_challenge_score
import pickle

class Training(object):
    def __init__(self, args):
        self.args = args
  
    def setup(self):
        '''Initializing the device conditions, datasets, dataloaders, 
        model, loss, criterion and optimizer
        '''
        
        # Consider the GPU or CPU condition
        if not torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = self.args.device_count
            self.args.logger.info('using {} gpu(s)'.format(self.device_count))
            assert self.args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            self.args.logger.info('using {} cpu'.format(self.device_count))

        # Load the datasets       
        training_set = ECGWindowPaddingDataset(self.args.train_path, window=self.args.window, nb_windows=self.args.nb_windows,
                                               filter_bandwidth=self.args.filter_bandwidth, all_features=self.args.all_features)
        self.train_dl = DataLoader(training_set,
                                   batch_size=self.args.batch_size,
                                   shuffle=True,
                                   num_workers=self.args.num_workers,
                                   pin_memory=(True if self.device == 'cuda' else False),
                                   drop_last=True)
        channels = training_set.channels

        validation_set = ECGWindowPaddingDataset(self.args.val_path, window=self.args.window, nb_windows=self.args.nb_windows,
                                               filter_bandwidth=self.args.filter_bandwidth, all_features=self.args.all_features)
        self.validation_files = validation_set.data
        self.val_dl = DataLoader(validation_set,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=self.args.num_workers,
                                 pin_memory=(True if self.device == 'cuda' else False),
                                 drop_last=True)

        self.model = CTN(in_channel=channels, out_channel=len(self.args.labels))

        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # If more than 1 CUDA device used, use data parallelism
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model) 
        
        # Optimizer
        self.optimizer = NoamOpt(256, 1, 4000, torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        self.model.to(self.device)
        
    def train(self):
        ''' PyTorch training loop
        '''
        
        self.args.logger.info('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
              (type(self.model).__name__, 
               type(self.optimizer).__name__,
               self.optimizer.optimizer.param_groups[0]['lr'], 
               self.args.epochs, 
               self.device))
        
        # Add all wanted history information
        history = {}
        history['train_loss'] = []
        history['train_micro_auroc'] = []
        history['train_micro_avg_prec'] = []  
        history['train_macro_auroc'] = []
        history['train_macro_avg_prec'] = [] 
        history['train_challenge_metric'] = []
        
        history['val_loss'] = []
        history['val_micro_auroc'] = []
        history['val_micro_avg_prec'] = []
        history['val_macro_auroc'] = []
        history['val_macro_avg_prec'] = []
        history['val_challenge_metric'] = []
        
        history['labels'] = self.args.labels
        history['epochs'] = self.args.epochs
        history['batch_size'] = self.args.batch_size
        history['lr'] = self.args.lr
        history['optimizer'] = self.optimizer
        history['train_csv'] = self.args.train_path
        history['val_csv'] = self.args.val_path
        
        start_time_sec = time.time()
        
        for epoch in range(1, self.args.epochs+1):
            
            # --- TRAIN ON TRAINING SET -----------------------------
            self.model.train()            
            train_loss = 0.0
            labels_all = torch.tensor((), device=self.device) # , device=torch.device('cuda:0')
            logits_prob_all = torch.tensor((), device=self.device)
            
            batch_loss = 0.0
            batch_count = 0
            step = 0

            patience = 5 # For early stopping
            patience_count = 0
            best_auroc = 0.0
            
            for batch_idx, (ecg_segs, feats_normalized, labels) in enumerate(self.train_dl):

                # Train instances use only one window
                ecg_segs = ecg_segs.transpose(1, 0)[0].float().to(self.device) # ECGs
                feats_normalized = feats_normalized.float().to(self.device) # metadata: age, gender and top handcrafted feautres
                labels = labels.float().to(self.device) # diagnoses

                self.optimizer.optimizer.zero_grad()
                logits = self.model(ecg_segs, feats_normalized)
                logits_prob = logits.sigmoid().data
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss_tmp = loss.item() * ecg_segs.size(0)

                labels_all = torch.cat((labels_all, labels), 0)
                logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)                    

                train_loss += loss_tmp

                loss.backward()
                self.optimizer.step()

                # self.args.logger.infoing training information
                if step % 100 == 0:
                    batch_loss += loss_tmp
                    batch_count += ecg_segs.size(0)
                    batch_loss = batch_loss / batch_count
                    self.args.logger.info('epoch {:^3} [{}/{}] train loss: {:>5.4f}'.format(
                        epoch, 
                        batch_idx * len(ecg_segs), 
                        len(self.train_dl.dataset), 
                        batch_loss
                    ))

                    batch_loss = 0.0
                    batch_count = 0
                step += 1

            train_loss = train_loss / len(self.train_dl.dataset)            
            train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc, train_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)

            # --- EVALUATE ON VALIDATION SET ------------------------------------- 
            self.model.eval()
            val_loss = 0.0  
            labels_all = torch.tensor((), device=self.device)
            logits_prob_all = torch.tensor((), device=self.device)  
            
            for ecg_segs, feats_normalized, labels in self.val_dl:
                ecg_segs = ecg_segs.float().to(self.device) # ECGs
                feats_normalized = feats_normalized.float().to(self.device) # age and gender
                labels = labels.float().to(self.device) # diagnoses 
                
                logits_tmp = []
                with torch.no_grad():  
                    # Loop over each window
                    for win in ecg_segs.transpose(1, 0):
                        logits = self.model(win, feats_normalized)
                        logits_tmp.append(logits)
                    
                    # Take the average of the sequence windows
                    logits = torch.stack(logits_tmp).mean(dim=0)
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                    logits_prob = logits.sigmoid().data
                    val_loss += loss.item() * ecg_segs.size(0)                                 
                    labels_all = torch.cat((labels_all, labels), 0)
                    logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

            val_loss = val_loss / len(self.val_dl.dataset)
            val_macro_avg_prec, val_micro_avg_prec, val_macro_auroc, val_micro_auroc, val_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
            
            # Create ROC Curves at the beginning, middle and end of training
            if epoch == 1 or epoch == self.args.epochs/2 or epoch == self.args.epochs:
                roc_curves(labels_all, logits_prob_all, self.args.labels, epoch, self.args.roc_save_dir)
  
            self.args.logger.info('epoch {:^4}/{:^4} train loss: {:<6.2f}  train macro auroc: {:<6.2f}  train challenge metric: {:<6.2f}'.format( 
                epoch, 
                self.args.epochs, 
                train_loss, 
                train_macro_auroc,
                train_challenge_metric))

            self.args.logger.info('                val loss:  {:<6.2f}   val macro auroc: {:<6.2f}    val challenge metric:  {:<6.2f}'.format(
                val_loss,
                val_macro_auroc,
                val_challenge_metric))
            
            # =====================================

            # Add information for training history
            history['train_loss'].append(train_loss)
            history['train_micro_auroc'].append(train_micro_auroc)
            history['train_micro_avg_prec'].append(train_micro_avg_prec)
            history['train_macro_auroc'].append(train_macro_auroc)
            history['train_macro_avg_prec'].append(train_macro_avg_prec)
            history['train_challenge_metric'].append(train_challenge_metric)
            
            history['val_loss'].append(val_loss)
            history['val_micro_auroc'].append(val_micro_auroc)
            history['val_micro_avg_prec'].append(val_micro_avg_prec)         
            history['val_macro_auroc'].append(val_macro_auroc)  
            history['val_macro_avg_prec'].append(val_macro_avg_prec)
            history['val_challenge_metric'].append(val_challenge_metric)

            # "During model training we monitored average AUC and used early stopping when validation AUC had stopped improving for 5 epochs."
            patience_count += 1

             # Save a model at every 5th epoch
            if epoch in list(range(self.args.epochs)[0::5]):
                self.args.logger.info('Saved model at the epoch {}!'.format(epoch))
                # Whether or not you use data parallelism, save the state dictionary this way
                # to have the flexibility to load the model any way you want to any device you want
                model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    
                # -- Save model
                model_savepath = os.path.join(self.args.model_save_dir,
                                              self.args.yaml_file_name + '_e' + str(epoch) + '.pth')
                torch.save(model_state_dict, model_savepath)

            
            # If validation macro-AUC is better than best AUC, save the model
            if epoch == self.args.epochs:

                self.args.logger.info('\nSaving the model, training history and validation logits...')
                    
                # Whether or not you use data parallelism, save the state dictionary this way
                # to have the flexibility to load the model any way you want to any device you want
                model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    
                # -- Save model
                model_savepath = os.path.join(self.args.model_save_dir,
                                              self.args.yaml_file_name + '.pth')
                torch.save(model_state_dict, model_savepath)

                # -- Save history
                history_savepath = os.path.join(self.args.model_save_dir,
                                                self.args.yaml_file_name + '_train_history.pickle')
                with open(history_savepath, mode='wb') as file:
                    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
                    

                # -- Save the logits from validation
                logits_csv_path = os.path.join(self.args.model_save_dir,
                                               self.args.yaml_file_name + '_val_logits.csv') 
                
                # Cleanup filenames to use as indexes
                cleanup_filenames = [os.path.basename(file) for file in self.validation_files]
                
                # Save the logits as a csv file where columns are the labels and 
                # indexes are the files which have been used in the validation phase
                logits_numpy = logits_prob_all.cpu().detach().numpy().astype(np.float32)
                logits_df = pd.DataFrame(logits_numpy, columns=self.args.labels, index=cleanup_filenames)
                logits_df.to_csv(logits_csv_path, sep=',')


        # END OF TRAINING LOOP        
        
        end_time_sec       = time.time()
        total_time_sec     = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / self.args.epochs
        self.args.logger.info('Time total:     %5.2f sec' % (total_time_sec))
        self.args.logger.info('Time per epoch: %5.2f sec' % (time_per_epoch_sec))