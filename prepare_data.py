import os, sys
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from utils import load_yaml

def lsdir(data_dir, suffix):
    '''Reading files from a given directory

    :param rootdir: Path of a directory
    :type rootdir: str
    :param suffix: A filename extension
    :type suffix: str

    :return file_list: All the filenames inside a given directory
    :rtype: list
    
    '''
    file_list = []
    assert os.path.exists(data_dir)
    
    for root, _, files in os.walk(data_dir):
       
        for file in files:
            if str(file).endswith(suffix):
                file_list.append(os.path.join(root, file))  
        
    return file_list


def reading_headerfiles(CT_codes_all, files, class_df):
    '''Updating an empty dataframe of files from a spesific source.
    Adding only the files which contain the diagnosis used as labels.
    Dataframe will include columns of filename, age, gender, fs and
    SNOMED CT codes included in the classification task. 

    :param CT_codes_all: List of all the SNOMED CT codes for the 
                         diagnoses included in the classification
    :type CT_codes_all: list
    :param files: List of all the files of a specific source
    :type files: list
    :param class_df: Zero-valued dataframe of size files x column names
    :type class_df: pandas.core.frame.DataFrame
    
    :return class_df: Dataframe of files of a spesific source
    :rype: pandas.core.frame.DataFrame
    '''
    
    i = -1
    for file in files:
        g = file.replace('.mat', '.hea')
        input_file_name = g
        flag = 1
        with open(input_file_name, 'r') as f:
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    tmp = [c.strip() for c in tmp]
                    if len(list(set(tmp).intersection(set(CT_codes_all)))) == 0:
                        flag = 0
        if flag == 1:
            i = i + 1
            class_df.loc[i, 'path'] = file

            with open(input_file_name, 'r') as f:
                for k, lines in enumerate(f):
                    if k == 0:
                        tmp = lines.split(' ')[2].strip()
                        class_df.loc[i, 'fs'] = int(tmp)
                    if lines.startswith('#Age'):
                        tmp = lines.split(': ')[1].strip()
                        if tmp == 'NaN':
                            class_df.loc[i, 'age'] = -1
                        else:
                            class_df.loc[i, 'age'] = int(tmp)
                    if lines.startswith('#Sex'):
                        tmp = lines.split(': ')[1].strip()
                        if tmp == 'NaN':
                            class_df.loc[i, 'gender'] = 'Unknown'
                        else:
                            class_df.loc[i, 'gender'] = tmp
                    if lines.startswith('#Dx'):
                        tmp = lines.split(': ')[1].split(',')
                        for c in tmp:
                            c = c.strip()
                            if c in CT_codes_all:
                                class_df.loc[i, c] = 1
    
    class_df = class_df.drop(class_df.index[i + 1:])
    return class_df


def stratified_shuffle_split(df, labels, n_split):
    ''' Splitting the data into training and validation sets
    using Multilabel Statified ShuffleSplit cross-validation.

    :param df: Dataframe of all the files from a specific source
    :type df: pandas.core.frame.Dataframe
    :param labels: List of train labels 
    :type labels: numpy.darray
    :param n_split: How many parts to divide the source into
    :type n_split: int

    :return train_csv: Training part of the split
    :return val_csv: Validation part of the split
    :rtype: list
    '''

    X = np.arange(labels.shape[0])
    msss = MultilabelStratifiedShuffleSplit(n_splits = n_split, train_size=0.75, test_size=0.25, random_state=2022)
    
    # Indexing split
    split_index_list = []
    for train_index, val_index in msss.split(X, labels):
        split_index_list.append([train_index, val_index])
        
        
    # Dividing into train and validation based on the indexes
    train_list = []
    val_list = []
    for i in range(len(split_index_list)):
        train_list.append(df.iloc[split_index_list[i][0], :])
        val_list.append(df.iloc[split_index_list[i][1], :])
    
    return train_list, val_list


def read_split_DB(data_directory, save_directory, labels):
    ''' Creating database-wise data splits and saving them in csv files

    :param data_directory: The location of the data files
    :type data_directory: str
    :param save_directory: The location where to save the csvs to
    :type save_directory: str
    :param labels: Labels in the classification
    :type labels: list
    '''

    # Preparing the directory to save the csv files to
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Getting directory names
    db_names = [directory for directory in os.listdir(data_directory) if not directory.startswith('.')]

    print('--Total of {} labels for the classification--'.format(len(labels)))
    
    # Iterating over databases for splits
    for db in db_names:
        
        # Path where to read the data from
        db_path = data_directory + db # './data/divided_sample/Georgia'
        
        # Getting files from the path
        file_names = lsdir(db_path, ".mat")
        
        # Putting all the samples into a dataframe
        columns_names = ['path', 'age', 'gender', 'fs'] + labels
        all_zeros = np.zeros((len(file_names), len(columns_names)))
        df_zeros = pd.DataFrame(all_zeros, columns=columns_names)
        
        # Excluding patients with diagnoses which are not included as labels in the classification
        df = reading_headerfiles(labels, file_names, df_zeros)
        
        # Saving database-wise splitted data into csvs
        df.to_csv(os.path.join(save_directory, '%s.csv' % (db)), sep=',', index=False)
        print('Created csv of the database {}!'.format(db))
        print('- Total of {} rows (excluded {} files since no wanted labels in them)'.format(len(df), len(file_names)-len(df)))
        print('-'*20)

        
def read_split_stratified(data_directory, save_directory, labels, train_val_splits):
    ''' Creating stratified data splits and saving them in csvs

    :param data_directory: The location of the data files
    :type data_directory: str
    :param save_directory: The location where to save the csvs to
    :type save_directory: str
    :param labels: Labels in the classification
    :type labels: list
    :param train_val_splits: Wanted train-test splits
    :type train_val_splits: dict
    '''
    
    # Preparing the directory to save the csv files to
    save_dir = save_directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
        
    print('--Total of {} labels for the classification--'.format(len(labels)))
    # All wanted splits to make
    stratified_splits = train_test_splits
    
    # Since maybe many splits in the directory at the same time!!
    n_all_splits = 0
    
    for i, data in enumerate(stratified_splits):
        
        train_split = data.train
        n_split = len(train_split)
        
        # All relative paths for different training data
        train_paths = [data_directory + db for db in train_split]

        # All the training files
        train_files = [lsdir(db_path, ".mat") for db_path in train_paths]
        train_files = [one_file for files in train_files for one_file in files]

        # Putting all the samples into a dataframe
        columns_names = ['path', 'age', 'gender', 'fs'] + labels
        all_zeros = np.zeros((len(train_files), len(columns_names)))
        df_zeros = pd.DataFrame(all_zeros, columns=columns_names)
        
        # Excluding patients with diagnoses which are not included as labels in the classification
        train_df = reading_headerfiles(labels, train_files, df_zeros)
        
        # Training labels - the last columns of the dataframe ()
        train_labels = train_df.loc[:, (train_df.sum(axis=0) != 0)].iloc[:, 4:].values
        
        # Stratifying the data
        train_tmp, val_tmp = stratified_shuffle_split(train_df, train_labels, n_split)
        
        # Saving the stratified splits to .csvs
        for i in range(n_split):
            train_tmp[i].to_csv(os.path.join(save_directory, '%s.csv' % ('train_split' + str(n_all_splits) + '_' + str(i))), sep=',', index=False)
            val_tmp[i].to_csv(os.path.join(save_directory, '%s.csv' % ('val_split' + str(n_all_splits) + '_' + str(i))), sep=',', index=False)
        
            if i == 0: # Need the test data for the stratified data
                test_data = data.test
                test_path = data_directory + test_data
                
                # Getting every test data file
                test_files = lsdir(test_path, ".mat")
                # Putting all the samples into a dataframe
                columns_names = ['path', 'age', 'gender', 'fs'] + labels
                all_zeros = np.zeros((len(test_files), len(columns_names)))
                df_zeros = pd.DataFrame(all_zeros, columns=columns_names)
                # Excluding diagnoses which are not included as labels in the classification
                test_df = reading_headerfiles(labels, test_files, df_zeros)
                test_df.to_csv(os.path.join(save_directory, '%s.csv' % ('test_split' + str(n_all_splits))), sep=',', index=False)
        
        n_all_splits += 1
        
        print('Created csv files for train-val-test split!')
        print('Training data was from the databases {}'.format(train_split))
        print('- Length of the first train_split is {}'.format(len(train_tmp[0])))
        print('- Length of the first val_split is {}'.format(len(val_tmp[0])))
        print('- {} rows excluded since no wanted labels in them'.format(len(train_files) - len(train_df)))
        print('Testing data was from the database {}'.format(test_data))
        print('- Length of the test_split is {}'.format(len(test_df)))
        print('- {} rows excluded since no wanted labels in them'.format(len(test_files) - len(test_df)))
        print('-'*20)
        
           
if __name__ == '__main__':
   
    # Parsing arguments
    if len(sys.argv) != 2:
        raise Exception('Include a yaml file as an argument, e.g., python prepare_data.py data.yaml')

    # Loading configurations from a yaml file
    yaml_file = sys.argv[1]
    yaml_filepath = os.path.join(os.getcwd(), 'configs', 'data_splitting', yaml_file)
    args = load_yaml(yaml_filepath)
    data_directory = args.data_dir
    save_directory = os.path.join(args.save_dir, yaml_file.split(".")[0]) 
    
    # Labels
    labels = ['426783006', '426177001', '164934002', '427084000', '164890007', '39732003', '164889003', '59931005', '427393009', '270492004']

    # Making either a stratified or a database-wise split
    if args.stratified:
        train_test_splits = args.splits
        read_split_stratified(data_directory, save_directory, labels, train_test_splits)
    else:
        read_split_DB(data_directory, save_directory, labels)
            
    print("Done.")
