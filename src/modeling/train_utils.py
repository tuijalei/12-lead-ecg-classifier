import os, sys
import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from .models.seresnet18 import resnet18
from .metrics import cal_multilabel_metrics, roc_curves
from ..dataloader.dataset import ECGDataset, get_transforms
import pickle

class Training(object):
    def __init__(self, args):
        self.args = args
  
    def setup(self):
        '''Initializing the device conditions, datasets, dataloaders, 
        model, loss, criterion and optimizer
        '''
        
        # Consider the GPU or CPU condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = self.args.device_count
            print('using {} gpu(s)'.format(self.device_count))
            assert self.args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            print('using {} cpu'.format(self.device_count))

        # Load the datasets       
        training_set = ECGDataset(self.args.train_path, 
                                  get_transforms('train', self.args.normalizetype, self.args.seq_length))
        validation_set = ECGDataset(self.args.val_path,
                                    get_transforms('val', self.args.normalizetype, self.args.seq_length)) 
        channels = training_set.channels
              
        self.train_dl = DataLoader(training_set,
                                   batch_size=self.args.batch_size,
                                   shuffle=True,
                                   num_workers=self.args.num_workers,
                                   pin_memory=(True if self.device == 'cuda' else False),
                                   drop_last=True)
        
        self.val_dl = DataLoader(validation_set,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=self.args.num_workers,
                                 pin_memory=(True if self.device == 'cuda' else False),
                                 drop_last=True)

        self.model = resnet18(in_channel=channels, 
                              out_channel=len(self.args.labels))

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)
        self.model.to(self.device)
        
    def train(self):
        ''' PyTorch training loop
        '''
        
        print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
              (type(self.model).__name__, 
               type(self.optimizer).__name__,
               self.optimizer.param_groups[0]['lr'], 
               self.args.epochs, 
               self.device))
        
        # Add all wanted history information
        history = {}
        history['train_loss'] = []
        history['train_micro_auroc'] = []
        history['train_micro_avg_prec'] = []  
        history['train_macro_auroc'] = []
        history['train_macro_avg_prec'] = [] 
        
        history['val_loss'] = []
        history['val_micro_auroc'] = []
        history['val_micro_avg_prec'] = []
        history['val_macro_auroc'] = []
        history['val_macro_avg_prec'] = []
        
        history['labels'] = self.args.labels
        history['epochs'] = self.args.epochs
        history['batch_size'] = self.args.batch_size
        history['lr'] = self.args.lr
        history['optimizer'] = self.optimizer
        history['criterion'] = self.criterion
        history['train_csv'] = self.args.train_path
        history['val_csv'] = self.args.val_path
        
        start_time_sec = time.time()
        
        for epoch in range(1, self.args.epochs+1):
            
            # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
            self.model.train()            
            train_loss = 0.0
            labels_all = torch.tensor((), device=self.device) # , device=torch.device('cuda:0')
            logits_prob_all = torch.tensor((), device=self.device)
            
            batch_loss = 0.0
            batch_count = 0
            step = 0
            
            for batch_idx, (ecgs, ag, labels) in enumerate(self.train_dl):
                ecgs = ecgs.to(self.device) # ECGs
                ag = ag.to(self.device) # age and gender
                labels = labels.to(self.device) # diagnoses in SMONED CT codes  
               
                with torch.set_grad_enabled(True):                    
        
                    logits = self.model(ecgs, ag) 
                    loss = self.criterion(logits, labels)
                    logits_prob = self.sigmoid(logits)      
                    loss_tmp = loss.item() * ecgs.size(0)
                    labels_all = torch.cat((labels_all, labels), 0)
                    logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)                    
                    
                    train_loss += loss_tmp
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # Printing the training information
                    if step % 100 == 0:
                        batch_loss += loss_tmp
                        batch_count += ecgs.size(0)
                        batch_loss = batch_loss / batch_count
                        print('epoch: {} [{}/{}], train loss: {:.4f}'.format(
                            epoch, 
                            batch_idx * len(ecgs), 
                            len(self.train_dl.dataset), 
                            batch_loss
                        ))

                        batch_loss = 0.0
                        batch_count = 0
                    step += 1

            
            train_loss = train_loss / len(self.train_dl.dataset)            
            train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc = cal_multilabel_metrics(labels_all, logits_prob_all)
            
            
            # --- EVALUATE ON VALIDATION SET ------------------------------------- 
            self.model.eval()
            val_loss = 0.0            
            labels_all = torch.tensor((), device=self.device)
            logits_prob_all = torch.tensor((), device=self.device)  
            
            for ecgs, ag, labels in self.val_dl:
                ecgs = ecgs.to(self.device) # ECGs
                ag = ag.to(self.device) # age and gender
                labels = labels.to(self.device) # diagnoses in SMONED CT codes 
                
                with torch.set_grad_enabled(False):                                                     
                      
                    logits = self.model(ecgs, ag)
                    loss = self.criterion(logits, labels)
                    logits_prob = self.sigmoid(logits)
                    val_loss += loss.item() * ecgs.size(0)                                 
                    labels_all = torch.cat((labels_all, labels), 0)
                    logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)
                    
            val_loss = val_loss / len(self.val_dl.dataset)
            val_macro_avg_prec, val_micro_avg_prec, val_macro_auroc, val_micro_auroc = cal_multilabel_metrics(labels_all, logits_prob_all)
            
            # Create ROC Curves at the beginning, middle and end of training
            if epoch == 1 or epoch == self.args.epochs/2 or epoch == self.args.epochs:
                roc_curves(labels_all, logits_prob_all, self.args.labels, epoch, self.args.roc_save_dir)
  
            print('epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                  (epoch, 
                   self.args.epochs, 
                   train_loss, 
                   train_micro_auroc, 
                   val_loss,
                   val_micro_auroc))
        
            # Add information for training history
            history['train_loss'].append(train_loss)
            history['train_micro_auroc'].append(train_micro_auroc)
            history['train_micro_avg_prec'].append(train_micro_avg_prec)
            history['train_macro_auroc'].append(train_macro_auroc)
            history['train_macro_avg_prec'].append(train_macro_avg_prec)
            
            history['val_loss'].append(val_loss)
            history['val_micro_auroc'].append(val_micro_auroc)
            history['val_micro_avg_prec'].append(val_micro_avg_prec)         
            history['val_macro_auroc'].append(val_macro_auroc)  
            history['val_macro_avg_prec'].append(val_macro_avg_prec)  
            
            # Save trained model after all the epochs
            if epoch == self.args.epochs:
                
                print('Saving the model...')

                model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                
                # Save model
                model_savepath = os.path.join(self.args.model_save_dir,
                                              self.args.yaml_file_name + '.pth')
                torch.save(model_state_dict, model_savepath)
                
                # Save history
                history_savepath = os.path.join(self.args.model_save_dir,
                                                self.args.yaml_file_name + '_history.pickle')
                with open(history_savepath, mode='wb') as file:
                    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)

            torch.cuda.empty_cache()
         
        # END OF TRAINING LOOP        
        
        end_time_sec       = time.time()
        total_time_sec     = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / self.args.epochs
        print()
        print('Time total:     %5.2f sec' % (total_time_sec))
        print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))