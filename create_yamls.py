import os, re, sys
from pathlib import Path
from ruamel.yaml import YAML
from itertools import combinations

def save_yaml(yaml_str, yaml_path, split):
    ''' Save the given string as a yaml file in the given location
    '''
    
    # Make the yaml directory
    if not os.path.isdir(yaml_path):
        os.mkdir(yaml_path)
    
    # Write the yaml file
    with open(os.path.join(yaml_path, split), 'w') as yaml_file:
        yaml = YAML()
        code = yaml.load(yaml_str)
        yaml.dump(code, yaml_file)
    
        
def create_test_yaml(test_csv, split, yaml_dict):
    ''' Make a yaml file for prediction. The base of it is presented above
    '''
    # The name of the model (the same as the name of the yaml file the model has been trained)
    model_name = split.split('.')[0] + '.pth'
    
    yaml_str = f"test_file: {test_csv}\nmodel: {model_name}\n"
    for key, value in yaml_dict.items():
        yaml_str += f"{key}: {value}\n"
    
    yaml_path = test_yaml_save_path
    save_yaml(yaml_str, yaml_path, split)
    

def create_train_yaml(train_csv, val_csv, split, yaml_dict):
    ''' Make a yaml file for training. The base of it is presented above
    '''
    
    yaml_str = f"train_file: {train_csv}\nval_file: {val_csv}\n"
    for key, value in yaml_dict.items():
        yaml_str += f"{key}: {value}\n"
    
    yaml_path = train_yaml_save_path
    save_yaml(yaml_str, yaml_path, split)


if __name__ == '__main__':

    # ------------------------------------------- # 
    # --- Parameters to create the yaml files --- #
    # ------------------------------------------- #

    # Create yamls using stratified data or database-wise (dbwise) splitted?
    stratified = True

    # NOTE: Change only the last folder name from the paths below.
    #       The modeling part is using the paths as they are ('/data/split_csvs/' and 'configs/<training/predicting>')

    # Only stratified: From where to load the csv files of original data split
    csv_path = os.path.join(os.getcwd(), 'data', 'split_csvs', '4fold_CVs')

    # Find data from the given csv path
    # Dbwise: Add csvs here to the second line if new databases used!
    data = sorted([file for file in os.listdir(csv_path) if not file.startswith('.') and file.endswith('.csv')]) \
        if stratified else ['PTB_PTBXL.csv', 'SPH.csv', 'G12EC.csv', 'ChapmanShaoxing_Ningbo.csv', 'CPSC_CPSC-Extra.csv']

    # --------------------------------------- #

    # Where to save the training yaml files
    train_yaml_save_path = os.path.join(os.getcwd(), 'configs', 'training', 'train_4fold')

    # Where to save the testing yaml files
    test_yaml_save_path = os.path.join(os.getcwd(), 'configs', 'predicting', 'predict_4fold')

    # How to name the yaml files (just the beginning; the name will be continued with ´_<number>´ [dbwise] or ´_<number>_<number>´ [stratified])
    name = 'split'

    # ------------------------------------- #
    # --- Parameters for the yaml files --- #
    # ------------------------------------- #

    # Don't change the keys! Train/val/test csv files as well as the model file 
    # (for testing) are automatically set in the yaml files

    train_dict = {
        
        # Directory where the csv file for data split are in 'data/split_cvs/'
        # (the same value as already set in `csv_path`, however, only the basename)
        'csv_path': os.path.basename(csv_path),
        'feature_path': 'smoke_features'

        # Training parameters
        'batch_size': 64,
        'num_workers': 0,
        'epochs': 50,
        'lr': 0.003,
        'weight_decay': 0.00001,

        # Device configurations
        'device_count': 1,

        # Decision threshold for predictions
        'threshold': 0.5,

        # For ECGs
        'bandwidth': [3, 45],
        'window': 15*500,
        'nb_windows': 5
    }

    test_dict = {
        
        # Directory where the csv file for data split are in 'data/split_cvs/'
        # (the same value as already set in `csv_path`, however, only the basename)
        'csv_path': os.path.basename(csv_path),

        # Device configurations
        'device_count': 1,

        # Decision threshold for predictions
        'threshold': 0.5,

        # For ECGs
        'bandwidth': [3, 45],
        'window': 15*500,
        'nb_windows': 5
    }

    # --------------------------------------- #
    # --------------------------------------- #

    # Names for the yaml files
    split_names = [] 

    # Train/val/test splitted csv files as a list of lists
    train_val_test = []

    # Setup either stratified yamls or dbwise yamls
    if stratified:

        # First, divide train, validation and test splits into own lists
        train_files = [file for file in data if 'train' in file]
        val_files = [file for file in data if 'val' in file]
        test_files = [file for file in data if 'test' in file]

        for i, pair in enumerate(list(zip(train_files, val_files))):

            # Training and validation files separately
            train_tmp, val_tmp = pair[0], pair[1]
            
            # Get the split number of the training file, used to name the corresponding yaml file
            split_num = re.search('_((\w*)_\d)', pair[0])
            split_names.append(str(split_num.group(1) + '.yaml'))
            
            train_split_num = split_num.group(2)
            for test_tmp in test_files:
                # Get the split number of the testing file
                test_split_num = re.search('_(\w*)', test_tmp).group(1)
                
                # If same split number in training, validation and prediction, combine
                if train_split_num == test_split_num:
                    train_val_test.append([train_tmp, val_tmp, test_tmp])

    else:

        # Find all combinations of the spesified data (= csv files of the databases)
        # One is left for testing so taking combinations in size of len(<all csv files>) -1
        for i, train_val_set in enumerate(combinations(data, len(data)-1)):
            test = next(file for file in data if not file in train_val_set)

            # And one is left for validation set so len(combinations took within first loop) -1
            for j, train_set in enumerate(combinations(train_val_set, len(train_val_set)-1)):
                val = next(file for file in data if not file in train_set and file != test)
                train_val_test.append([list(train_set), val, test])
                split_names.append(name + '_' + str(i+1) + '_' + str(j+1) + '.yaml')

    print('Total of {} training, validation and testing sets'.format(len(train_val_test)))
    print('First two training, validation and testing pairs')
    print(*train_val_test[:2], sep='\n') 
    print()

    for pair, split in list(zip(train_val_test, split_names)):
        train_tmp, val_tmp, test_tmp = pair[0], pair[1], pair[2]

        print('Training, validation and testing sets are'.format())
        print(train_tmp, '\t', val_tmp, '\t', test_tmp)
        print('Yaml file will be named as', split)
        print()

        # Training yaml file
        create_train_yaml(train_tmp, val_tmp, split, train_dict)
        
        # Testing yaml file
        create_test_yaml(test_tmp, split, test_dict)