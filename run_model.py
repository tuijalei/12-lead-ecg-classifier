import numpy as np, os, sys
import torch
import random
import pandas as pd
from utils import load_yaml
from src.modeling.predict_utils import Predicting
import logging

def read_yaml(file, model_save_dir='', multiple=False):
    ''' Read a given yaml and perform classification predictions.
    Evaluate the predictions.
    
    :param file (str): Absolute path for the yaml file wanted to read
    :param model_save_dir (str): If multiple yamls are read, the model directory is  
                                 a subdirectory of the 'experiments' directory
    :param multiple (boolean): Check if multiple yamls are read
    '''

    # Seed
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Load yaml
    print('Loading arguments from', os.path.basename(file))
    args = load_yaml(file)

    # Path where the needed CSV file exists
    csv_root = os.path.join(os.getcwd(), 'data', 'split_csvs', args.csv_path)
    
    # Update paths
    args.test_path = os.path.join(csv_root, args.test_file)
    args.yaml_file_name = os.path.splitext(file)[0]
    args.yaml_file_name = os.path.basename(args.yaml_file_name)
    
    # Output directory based on if multiple yaml files are run or only one
    args.output_dir = os.path.join(os.getcwd(),'experiments', model_save_dir, args.yaml_file_name) if multiple else os.path.join(os.getcwd(),'experiments', args.yaml_file_name)
    
    # Make a subdirectory into the output directory where to save the predictions
    args.pred_save_dir = os.path.join(args.output_dir, 'predictions')
    
    # Make sure the directory for predictions and other outputs exists
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.pred_save_dir, exist_ok=True)

    # Find the trained model from the ´experiments´ directory as it should be saved there
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'experiments')):
        if args.model in files:
            args.model_path = os.path.join(root, args.model)
    
    # Check if model_path never set, i.e., the trained model was found
    if not hasattr(args, 'model_path'):
        raise AttributeError('No path found for the model. Check if you have trained one.')

    # Load labels
    args.labels = pd.read_csv(args.test_path, nrows=0).columns.tolist()[4:]

    # For logging purposes
    logs_path = os.path.join(args.output_dir, args.yaml_file_name + '_predict.log')
    logging.basicConfig(filename=logs_path, 
                        format='%(asctime)s %(message)s', 
                        filemode='w',
                        datefmt='%Y-%m-%d %H:%M:%S') 
    args.logger = logging.getLogger(__name__) 
    args.logger.setLevel(logging.DEBUG) 
    
    args.logger.info('Arguments:')
    args.logger.info('-'*10)
    for k, v in args.__dict__.items():
        args.logger.info('{}: {}'.format(k, v))
    args.logger.info('-'*10) 

    args.logger.info('Making predictions...')

    pred = Predicting(args)
    pred.setup()
    pred.predict()

    
def read_multiple_yamls(path):
    ''' Read multiple yaml files from the given directory
    
    :param directory (str): Absolute path for the directory
    '''
    # All yaml files
    yaml_files = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    
    # Save all trained models in the same subdirectory in the 'experiments' directory
    dir_name = os.path.basename(path)
    model_save_dir = os.path.join(os.getcwd(),'experiments', dir_name) 

    # Running the yaml files and training models for each
    for file in yaml_files:
        read_yaml(file, model_save_dir, True)


if __name__ == '__main__':

    # Load args
    given_arg = sys.argv[1]
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