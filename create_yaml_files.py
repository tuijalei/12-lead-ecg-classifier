import os, re, sys
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
    
        
def write_yaml(csvs, split, yaml_dict, phase):
    ''' Make a yaml file for prediction. The base of it is presented above

    :csvs [list]: for training, there can be two csv in a list of csvs, 
                  where the first corresponds to a training csv and the second to a validation csv
                  for testing, there needs to be only one csv file listed
    '''
    yaml_str = ''

    if phase == 'train':
        yaml_str = f"train_file: {csvs[0]}\nval_file: {csvs[1]}\n"
        yaml_path = train_yaml_save_path
    
    elif phase == 'test':
        # The name of the model (the same as the name of the yaml file the model has been trained)
        model_name = split.split('.')[0] + '.pth'
        yaml_str = f"test_file: {csvs[0]}\nmodel: {model_name}\n"
        yaml_path = test_yaml_save_path

    if yaml_str == '':
        raise Exception('No string to write to a YAML file.')

    for key, value in yaml_dict.items():
        if isinstance(value, float):
            yaml_str += f"{key}: {value:f}\n"
        else:
            yaml_str += f"{key}: {value}\n"

    save_yaml(yaml_str, yaml_path, split)


if __name__ == '__main__':

    # ------------------------------------------- # 
    # --- Parameters to create the yaml files --- #
    # ------------------------------------------- #

    # Create yamls using stratified (kfold or shufflesplit) data or database-wise (dbwise) splitted?
    stratified = True 

    # NOTE: Change only the last folder name from the paths below.
    #       The modeling part is using the paths as they are ('/data/split_csvs/' and 'configs/<training/predicting>')

    # From where to load the csv files of original data split
    csv_path = os.path.join(os.getcwd(), 'data', 'split_csvs', 'stratified_smoke')

    # Find data from the given csv path
    # Dbwise: Add the names of the CSV files
    data = sorted([file for file in os.listdir(csv_path) if not file.startswith('.') and file.endswith('.csv')]) \
        if stratified else ['PTB_PTBXL.csv', 'SPH.csv', 'G12EC.csv', 'ChapmanShaoxing_Ningbo.csv', 'CPSC_CPSC-Extra.csv']
    
    data_dirs = sorted([d for d in os.listdir(csv_path) if os.path.isdir(os.path.join(csv_path, d))])

    # --------------------------------------- #

    # Where to save the training yaml files
    train_yaml_save_path = os.path.join(os.getcwd(), 'configs', 'training', 'train_yamls_smoke')

    # Where to save the testing yaml files
    test_yaml_save_path = os.path.join(os.getcwd(), 'configs', 'predicting', 'test_yamls_smoke')

    # How to name the yaml files 
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

        # Training parameters
        'batch_size': 10,
        'num_workers': 0,
        'epochs': 1,
        'lr': 0.003,
        'weight_decay': 0.00001,

        # Device configurations
        'device_count': 1,

        # Decision threshold for predictions
        'threshold': 0.5,

        # For ECGs
        'bandwidth': ''
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
        'bandwidth': ''
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
            test_tmp = next(file for file in data if not file in train_val_set)

            # And one is left for validation set so len(combinations took within first loop) -1
            for j, train_tmp in enumerate(combinations(train_val_set, len(train_val_set)-1)):
                val_tmp = next(file for file in data if not file in train_tmp and file != test_tmp)

                # Convert a list of csv file names into a single name for a csv file
                # e.g. from ['G12EC.csv', 'PTB_PTBXL.csv', 'SPH.csv'] to G12EC_PTB_PTBXL_SPH.csv
                train_tmp = [db.split('.')[0] for db in train_tmp]
                train_tmp = '_'.join(sorted(train_tmp, key=str.lower)) + '.csv'
                
                train_val_test.append([train_tmp, val_tmp, test_tmp])
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
        write_yaml([train_tmp, val_tmp], split, train_dict, phase='train')
        
        # Testing yaml file
        write_yaml([test_tmp], split, test_dict, phase='test')