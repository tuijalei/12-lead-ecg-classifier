import os
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

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
    assert os.path.exists(data_dir), 'Check the path for data directory'
    
    for root, _, files in os.walk(data_dir):
       
        for file in files:
            if str(file).endswith(suffix):
                file_list.append(os.path.join(root, file))  
        
    return file_list


def read_headerfiles(CT_codes_all, files, class_df):
    '''Filling an empty dataframe with files from a spesific source.
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


def dbwise_csvs(data_directory, save_directory, labels):
    ''' Creating database-wise data splits and saving them in csv files

    :param data_directory: The location of the data files
    :type data_directory: str
    :param save_directory: The location where to save the csvs to
    :type save_directory: str
    :param labels: Labels in the classification
    :type labels: list
    '''

    # Preparing the directory where to save the csv files
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Getting directory names
    db_names = [directory for directory in os.listdir(data_directory) if not directory.startswith('.')]

    print('--Total of {} labels for the classification--'.format(len(labels)))
    
    # Iterating over databases for splits
    for db in db_names:
        
        # Absolute path where to read the data from
        db_path = os.path.join(data_directory, db)
        
        # Getting files from the path
        file_names = lsdir(db_path, ".mat")
        
        # Putting all the ECGs into a dataframe: first, create an empty one
        columns_names = ['path', 'age', 'gender', 'fs'] + labels
        all_zeros = np.zeros((len(file_names), len(columns_names)))
        df_zeros = pd.DataFrame(all_zeros, columns=columns_names)
        
        # Excluding patients with diagnoses which are not included as labels for classification
        # Filling the empty dataframe
        ecg_df = read_headerfiles(labels, file_names, df_zeros)
        
        # Saving database-wise splitted data into csvs
        ecg_df.to_csv(os.path.join(save_directory, '%s.csv' % (db)), sep=',', index=False)
        
        print('Created csv of the database {}!'.format(db))
        print('- Total of {} rows (excluded {} files as no wanted labels in them)'.format(len(ecg_df), len(file_names)-len(ecg_df)))
        print('-'*20)

        
def stratified_csvs(data_directory, save_directory, labels, train_test_splits):
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
    
    # Preparing the directory where to save the csv files
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) 
        
    print('--Total of {} labels for the classification--'.format(len(labels)))
    
    # Iterate over training and validation splits
    for split, data in train_test_splits.items():
        
        # Get databases used for training data
        train_data = data['train']

        # Number of databases used for training data
        n_split = len(train_data)
        
        # All relative paths for different training data(bases)
        train_paths = [os.path.join(data_directory, db) for db in train_data]

        # All the training files
        train_files = [lsdir(db_path, ".mat") for db_path in train_paths]
        train_files = [one_file for files in train_files for one_file in files]

        # Putting all the ECGs into a dataframe: first, create an empty one
        columns_names = ['path', 'age', 'gender', 'fs'] + labels
        all_zeros = np.zeros((len(train_files), len(columns_names)))
        df_zeros = pd.DataFrame(all_zeros, columns=columns_names)
        
        # Excluding patients with diagnoses which are not included as labels for classification
        # Filling the empty dataframe
        ecg_df = read_headerfiles(labels, train_files, df_zeros)
        
        # Training labels - the last columns of the dataframe
        train_labels = ecg_df.loc[:, (ecg_df.sum(axis=0) != 0)].iloc[:, 4:].values
        
        # Stratifying the data
        train_sets, val_sets = stratified_shuffle_split(ecg_df, train_labels, n_split)
        
        # Saving the stratified training and validation splits to csv files
        for i in range(n_split):
            train_sets[i].to_csv(os.path.join(save_directory, '%s.csv' % ('train_' + split + '_' + str(i+1))), sep=',', index=False)
            val_sets[i].to_csv(os.path.join(save_directory, '%s.csv' % ('val_' + split + '_' + str(i+1))), sep=',', index=False)

            if i == 0: # Save the test data only once
                test_data = data['test']
                test_path = os.path.join(data_directory, test_data)
                
                # Getting every test data file
                test_files = lsdir(test_path, ".mat")
                
                # Putting all the samples into a dataframe
                columns_names = ['path', 'age', 'gender', 'fs'] + labels
                all_zeros = np.zeros((len(test_files), len(columns_names)))
                df_zeros = pd.DataFrame(all_zeros, columns=columns_names)
                
                # Excluding diagnoses which are not included as labels in the classification
                # Filling the empty dataframe
                test_set = read_headerfiles(labels, test_files, df_zeros)
                test_set.to_csv(os.path.join(save_directory, '%s.csv' % ('test_' + split)), sep=',', index=False)
        
        print('Created csv files for train-val-test split!')
        print('Training data was from the databases {}'.format(train_data))
        print('- Length of the first train_split is {}'.format(len(train_sets[0])))
        print('- Length of the first val_split is {}'.format(len(val_sets[0])))
        print('- {} rows excluded as no wanted labels in them'.format(len(train_files) - len(ecg_df)))
        print('Testing data was from the database {}'.format(test_data))
        print('- Length of the test_split is {}'.format(len(test_set)))
        print('- {} rows excluded as no wanted labels in them'.format(len(test_files) - len(test_set)))
        print('-'*20)


if __name__ == '__main__':
    ''' The scipt to create csv files for training and testing.
    Note that with this script, you make the decision about the labels
    which you want to use in classification.
    
    Consider the following parameters:
    ------------------------------------------
    
        :param stratified: perform either a database-wise or a stratified data split
        :type stratified: boolean
        :param data_dir: where to load the ecgs and header files from
        :type data_dir: str
        :param csv_dir: where to save the 
        :type csv_dir: str
        :param labels: wanted labels to include in classification, must be in SNOMED CT Codes
        :type labels: list

        
    '''

    # ----- WHICH DATA SPLIT DO YOU WANT TO USE WHEN CREATING CSV FILES?
    # Database-wise split :: stratified = False
    # Stratified split :: stratified = True
    stratified = True
    
    # ----- WHERE TO LOAD THE ECGS FROM - give the name of the data directory
    # Note that the root for this is the 'data' dictionary
    data_dir = 'physionet_preprocessed_smoke'
    
    # ----- WHERE TO SAVE THE CSV FILES - give a name for the new directory
    # Note that the root for this is the 'data/split_csv/' directory
    csv_dir = 'physionet_stratified_smoke'

    # ----- LABELS TO USE IN SNOMED CT CODES: THESE ARE USED FOR CLASSIFICATION 
    labels = ['426783006', '426177001', '164934002', '427084000', '164890007', '39732003', '164889003', '59931005', '427393009', '270492004']
   
    
    # -----------------------------------------------------------------------
    # ----- STRATIFIED DATA SPLIT
    if stratified:
        
         # Data directory from where to read data for csvs
        data_dir =  os.path.join(os.getcwd(), 'data', data_dir)
        
        # Directory to save created csv files
        csv_dir =  os.path.join(os.getcwd(), 'data', 'split_csvs', csv_dir)

        # Splits to divide the data into when making a stratified split
        # The other splits are represented later in comments
        train_test_splits = {
            'split_1': {    
                'train': ['G12EC', 'INCART', 'PTB_PTBXL', 'ChapmanShaoxing_Ningbo'],
                'test': 'CPSC_CPSC-Extra'
            },
            'split_2': {    
                    'train': ['G12EC', 'INCART', 'PTB_PTBXL', 'CPSC_CPSC-Extra'],
                    'test': 'ChapmanShaoxing_Ningbo'
                },
            'split_3': {    
                    'train': ['G12EC', 'INCART', 'CPSC_CPSC-Extra', 'ChapmanShaoxing_Ningbo'],
                    'test': 'PTB_PTBXL'
                },
            'split_4': {    
                    'train': ['G12EC', 'PTB_PTBXL', 'CPSC_CPSC-Extra', 'ChapmanShaoxing_Ningbo'],
                    'test': 'INCART'
                },
            'split_5': {    
                    'train': ['INCART', 'PTB_PTBXL', 'CPSC_CPSC-Extra', 'ChapmanShaoxing_Ningbo'],
                    'test': 'G12EC'
                }
        }

        # Perform stratified data split
        stratified_csvs(data_dir, csv_dir, labels, train_test_splits)
    
    # ----- DATABASE-WISE DATA SPLIT
    else:
        
        # Data directory from where to read data for csvs
        data_dir =  os.path.join(os.getcwd(), 'data', data_dir)
        
        # Directory to save created csv files
        csv_dir =  os.path.join(os.getcwd(), 'data', 'split_csvs', csv_dir)

        # Perform database-wise data split
        dbwise_csvs(data_dir, csv_dir, labels)
    # -----------------------------------------------------------------------

    print("Done.")
    
    # ----------------------------------------
    # Different train-test splits for the Physionet Challenge 2021 data:
    # (you can use these by just adding them to the 'train_test_splits' dictionary)
    # ----------------------------------------
    #'split_1': {    
    #        'train': ['G12EC', 'INCART', 'PTB_PTBXL', 'ChapmanShaoxing_Ningbo'],
    #        'test': 'CPSC_CPSC-Extra'
    #    },
    # 'split_2': {    
    #        'train': ['G12EC', 'INCART', 'PTB_PTBXL', 'CPSC_CPSC-Extra'],
    #        'test': 'ChapmanShaoxing_Ningbo'
    #    },
    # 'split_3': {    
    #        'train': ['G12EC', 'INCART', 'CPSC_CPSC-Extra', 'ChapmanShaoxing_Ningbo'],
    #        'test': 'PTB_PTBXL'
    #    },
    # 'split_4': {    
    #        'train': ['G12EC', 'PTB_PTBXL', 'CPSC_CPSC-Extra', 'ChapmanShaoxing_Ningbo'],
    #        'test': 'INCART'
    #    },
    # 'split_5': {    
    #        'train': ['INCART', 'PTB_PTBXL', 'CPSC_CPSC-Extra', 'ChapmanShaoxing_Ningbo'],
    #        'test': 'G12EC'
    #    },
    # ----------------------------------------
