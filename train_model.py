import torch
import numpy as np
import random    
import sys
import os
import pandas as pd
from utils import load_yaml
from src.modeling.train_utils import Training

def read_yaml(file, model_save_dir='', multiple=False):
    ''' Read a yaml file and perform training.
    
    :param file: Absolute path for the yaml file wanted to read
    :type file: str
    :param model_save_dir: If multiple yamls are read, the model directory is  
                           a subdirectory of the 'experiments' directory
    :type model_save_dir: str
    :param multiple: Check if multiple yamls are read
    :type multiple: boolean
    '''
    
    # Load yaml
    args = load_yaml(file)

    # Update paths
    args.train_path = os.path.join(csv_root, args.train_file)
    args.val_path = os.path.join(csv_root, args.val_file)
    args.yaml_file_name = os.path.splitext(file)[0]
    args.yaml_file_name = os.path.basename(args.yaml_file_name)
    
    if multiple:
        args.model_save_dir = model_save_dir
        args.roc_save_dir = os.path.join(os.getcwd(),'experiments', model_save_dir, 'ROC_' + args.yaml_file_name)
    else:
        args.model_save_dir = os.path.join(os.getcwd(),'experiments', args.yaml_file_name)
        args.roc_save_dir = os.path.join(os.getcwd(),'experiments', args.yaml_file_name, 'ROC_curves')

    # Parameters --------------------
    args.device_count = 1
    args.lr = 0.003
    args.weight_decay = 0.00001
    
    # Transforms arguments ----------
    args.seq_length = 4096
    args.normalizetype = '0-1'
    # -------------------------------
    
    # Get labels from the train csv
    # The class labels start from the 4th index (exclude path, age, gender and fs)
    args.labels = pd.read_csv(args.train_path, nrows=0).columns.tolist()[4:]

    # Directory for training information
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    
    # Directory for ROCs
    if not os.path.exists(args.roc_save_dir):
        os.makedirs(args.roc_save_dir)
        
    print('Arguments:')
    for k, v in args.__dict__.items():
        print(k + ':', v)
    
    print('-'*10)  

    print('Running training code...')

    trainer = Training(args)
    trainer.setup()
    trainer.train() 
    

def read_multiple_yamls(path):
    ''' Read multiple yaml files from the given directory
    
    :param directory: Absolute path for the directory
    :type path: str
    '''
    # All yaml files
    yaml_files = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    
    # Save all trained models in the same subdirectory in the 'experiments' directory
    dir_name = os.path.basename(path)
    model_save_dir = os.path.join(os.getcwd(),'experiments', dir_name) 
    
    # Reading the yaml files and training models for each
    for file in yaml_files:
        read_yaml(file, model_save_dir, True)


if __name__ == '__main__':

    # Seed
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic =  True
    
    # Paths
    csv_root = './data/split_csvs/physionet_stratified_smoke/'
    data_root = './data/physionet_preprocessed_smoke/' 

    # Load args
    given_arg = sys.argv[1]
    print('Loading args from:', given_arg)
    arg_path = os.path.join(os.getcwd(), 'configs', 'training', given_arg)

    # Check if a yaml file or a directory given as an argument
    # Possible multiple yamls for training phase!
    if os.path.exists(arg_path):

        if 'yaml' in given_arg:
            # Run one yaml
            read_yaml(arg_path)
        else:
            # Run multiple yamls from a directory
            read_multiple_yamls(arg_path)
            
    else:
        raise Exception('No such file nor directory exists! Check the arguments.')

    print('Done.')