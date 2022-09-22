import numpy as np, os, sys
import torch
import random
import pandas as pd
from utils import load_yaml
from src.modeling.predict_utils import Predicting
from src.modeling.metrics import evaluate_predictions


def read_yaml(file, model_save_dir='', multiple=False):
    ''' Read a given yaml and perform classification predictions.
    Evaluate the predictions made.
    
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
    args.test_path = os.path.join(csv_root, args.test_file)
    args.yaml_file_name = os.path.splitext(file)[0]
    args.yaml_file_name = os.path.basename(args.yaml_file_name)
    
    # Output directory based on if multiple yaml files are run or only one
    args.output_dir = os.path.join(os.getcwd(),'experiments', model_save_dir, args.yaml_file_name) if multiple else os.path.join(os.getcwd(),'experiments', args.yaml_file_name)
    
    # Make sure the directory for outputs exists
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)    

    # Find the trained model from the ´experiments´ directory
    # since it should be saved there
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'experiments')):
        if args.model in files:
            args.model_path = os.path.join(root, args.model)

    # Device count
    args.device_count = 2
    # Transforms --------------------
    args.seq_length = 4096
    args.normalizetype = 'none'
    # -------------------------------

    # Load labels
    args.labels = pd.read_csv(args.test_path, nrows=0).columns.tolist()[4:]

    print('Arguments:\n' + '-'*10)
    for k, v in args.__dict__.items():
        print(k + ':', v)
    print('-'*10) 

    print('Making predictions...')

    pred = Predicting(args)
    pred.predict()

    print()
    print('Evaluating predictions...')

    evaluate_predictions(args.test_path, args.output_dir)

    
def read_multiple_yamls(path):
    ''' Read multiple yaml files from the given directory
    
    :param directory: Absolute path for the directory
    :type path: str
    '''
    # All yaml files
    yaml_files = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    
    # Save all trained models in the same directory in the 'experiments' directory
    dir_name = os.path.basename(path)
    model_save_dir = os.path.join(os.getcwd(),'experiments', dir_name) 
    
    # Running the yaml files and training models for each
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
    torch.backends.cudnn.deterministic = True

    # Paths
    csv_root = './data/split_csvs/physionet_DBwise/'
    data_root = './data/physionet_preprocessed/'

    # Load args
    given_arg = sys.argv[1]
    print('Loading arguments from', given_arg)
    arg_path = os.path.join(os.getcwd(), 'configs', 'predicting', given_arg)

    # Check if a yaml file or a directory given as an argument
    # Possible multiple yamls for prediction and evaluation phase!
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