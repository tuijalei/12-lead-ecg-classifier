import numpy as np, os, sys
import torch
import random
import pandas as pd
from utils import load_yaml
from src.modeling.predict_utils import Predicting

def read_yaml(file, csv_root, model_save_dir='', multiple=False):
    ''' Read a given yaml and perform classification predictions.
    Evaluate the predictions.
    
    :param file: Absolute path for the yaml file wanted to read
    :type file: str
    :param csv_root: Absolute path for the csv file
    :type csv_root: str
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
        
    # Make a subdirectory into the output directory where to save the predictions
    args.pred_save_dir = os.path.join(args.output_dir, 'predictions')
    
    # Make sure the directory for predictions exists
    if not os.path.isdir(args.pred_save_dir):
        os.makedirs(args.pred_save_dir)    

    # Find the trained model from the ´experiments´ directory as it should be saved there
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'experiments')):
                if args.model in files:
                    args.model_path = os.path.join(root, args.model)
    
    # Check if model_path never set, i.e., the trained model was found
    try:
        args.model_path
    except AttributeError as ne:
        print('AttributeError:', ne, 'I.e. model not found. Check if you´ve trained one.')

    # Load labels
    args.labels = pd.read_csv(args.test_path, nrows=0).columns.tolist()[4:]
    
    print('Arguments:\n' + '-'*10)
    for k, v in args.__dict__.items():
        print(k + ':', v)
    print('-'*10) 

    print('Making predictions...')

    pred = Predicting(args)
    pred.setup()
    pred.predict()

    
def read_multiple_yamls(path, csv_root):
    ''' Read multiple yaml files from the given directory
    
    :param directory: Absolute path for the directory
    :type path: str
    '''
    # All yaml files
    yaml_files = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    
    # Save all trained models in the same subdirectory in the 'experiments' directory
    dir_name = os.path.basename(path)
    model_save_dir = os.path.join(os.getcwd(),'experiments', dir_name) 

    # Running the yaml files and training models for each
    for file in yaml_files:
        read_yaml(file, csv_root, model_save_dir, True)


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

    # ----- Set the path here! -----
    
    # Root where the needed CSV file exists
    csv_root = os.path.join(os.getcwd(), 'data', 'split_csvs', 'physionet_stratified_smoke')
    
    # ------------------------------

    # Load args
    given_arg = sys.argv[1]
    print('Loading arguments from', given_arg)
    arg_path = os.path.join(os.getcwd(), 'configs', 'predicting', given_arg)
    
    # Check if a yaml file or a directory given as an argument
    # Possible multiple yamls for prediction and evaluation phase!
    if os.path.exists(arg_path):

        if 'yaml' in given_arg:
            # Run one yaml
            read_yaml(arg_path, csv_root)
        else:
            # Run multiple yamls from a directory
            read_multiple_yamls(arg_path, csv_root)

    else:
        raise Exception('No such file nor directory exists! Check the arguments.')

    print('Done.')