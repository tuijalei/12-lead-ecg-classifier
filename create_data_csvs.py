import os, sys, glob
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from itertools import combinations

def lsdir(data_dir):
    '''Find the files from the given directory.

    :param rootdir: Path of a directory
    :type rootdir: str

    :return file_list: All the filenames inside the given directory.
                       Names are absolute paths. 
    :rtype: list
    
    '''
    file_list = []
    assert os.path.exists(data_dir), 'Check the path for data directory'

    # Find only the files with the spesific suffix
    wanted_suffixes = ['.mat', '.h5']
    
    for root, _, files in os.walk(data_dir):
       
        for file in files:

            # Check if the file has the spesified suffix and append to the list only if it does
            if any(str(file).endswith(suffix) for suffix in wanted_suffixes):
                file_list.append(os.path.join(root, file))  
        
    return file_list


def read_metacsv(CT_codes_all, files, columns, metacsv):
    '''Find information of age, gender, sample frequency and diagnoses from
    csv files. Return the information in a list-of-dictionaries which is 
    later concatenated to a dataframe.

    :param CT_codes_all: List of all the SNOMED CT codes for the 
                         diagnoses included in the classification
    :type CT_codes_all: list
    :param files: List of all the files of a specific source
    :type files: list
    :param columns: Columns of the ECG dataframe to which the information
                    is gathered.
    :type columns: list
    
    :return metadata_rows: List of dictionaries of the information
    :rype: list-of-dictionaries
    '''
    metadata_rows = []

    # Read the metadata from the given csv file
    metacsv_df = pd.read_csv(metacsv)

    for file in files:

        # The basename of the file
        file_name = os.path.basename(file)

        # Gather all information to a dictionary
        metadata_dict = {key: None for key in columns}
        
        # ECG id should be found in the csv file: Check if the given file is named there
        if file_name in metacsv_df['ECG_ID'].values:

            # Get the index from where to read the other informations
            row_idx = metacsv_df.index[metacsv_df['ECG_ID'] == file_name].tolist()
            
            # Find the diagnoses of the given file
            dx = metacsv_df.loc[row_idx, 'Dx'].values[0]
            dx = [c.strip() for c in dx.split(',')]

            # Check whether the diagnoses are included within SNOMED CT Codes
            # If yes, gather the metadata
            if bool(set.intersection(set(dx), set(CT_codes_all))):
                
                # Mark diagnoses that are found with the value of 1 in the corresponding column
                for code in dx:
                    if code in CT_codes_all:
                        metadata_dict[code] = 1
                
                # Add zero to all other diagnoses
                for key in metadata_dict.keys():
                    if not metadata_dict[key]:
                        metadata_dict[key] = 0

                # Add a path of the file
                metadata_dict['path'] = file

                # Find the sample frequency
                fs = metacsv_df.loc[row_idx, 'fs']
                metadata_dict['fs'] = int(fs)
                
                # Find the age information
                age = metacsv_df.loc[row_idx, 'Age']
                if str(age) == 'NaN':
                    metadata_dict['age'] = -1
                else:
                    metadata_dict['age'] = int(age)
                
                # Find the gender information
                gender = metacsv_df.loc[row_idx, 'Sex'].values[0]
                if str(gender) == 'NaN':
                    metadata_dict['gender'] = 'Unknown'
                else:
                    metadata_dict['gender'] = str(gender)

            metadata_rows.append(metadata_dict) 

    return metadata_rows

def read_headerfiles(CT_codes_all, files, columns):
    '''Find information of age, gender, sample frequency and diagnoses from
    header files. Return the information in a list-of-dictionaries which is 
    later concatenated to a dataframe.

    :param CT_codes_all: List of all the SNOMED CT codes for the 
                         diagnoses included in the classification
    :type CT_codes_all: list
    :param files: List of all the files of a specific source
    :type files: list
    :param columns: Columns of the ECG dataframe to which the information
                    is gathered.
    :type columns: list
    
    :return metadata_rows: List of dictionaries of the information
    :rype: list-of-dictionaries
    '''

    # Gather all information to a list-of-dictionaries
    metadata_rows = []

    # Iterate over files
    for file in files:

        # Each ECG mat file should have a corresponding hea file
        input_file_name = file.replace('.mat', '.hea')

        # Gather all information to a dictionary
        metadata_dict = {key: None for key in columns}

        # Flag to mark if diagnoses in a file found from the SNOMED CT Codes used as labels
        labels_found = False

        # Read the given file
        with open(input_file_name, 'r') as f:

            # Iterate over lines to first check the diagnoses
            for lines in f:

                # Find the diagnoses from the line that starts with '#Dx'
                if lines.startswith('#Dx'):
                    dx = lines.split(': ')[1]
                    dx = [c.strip() for c in dx.split(',')]

                    # If any diagnosis is found among the SNOMED CT Codes, read metadata
                    # and fill up the metadata dictionary
                    if bool(set.intersection(set(dx), set(CT_codes_all))):
                        labels_found = True
            
            # If the diagnoses that are used as labels are found,
            # we need to gather the metadata from the header file
            if labels_found:

                # Mark diagnoses that are found with the value of 1 in the corresponding column
                for code in dx:
                    if code in CT_codes_all:
                        metadata_dict[code] = 1

                # Add zero to all other diagnoses
                for key in metadata_dict.keys():
                    if not metadata_dict[key]:
                        metadata_dict[key] = 0
 
                # Move back to the beginning of the file
                f.seek(0)

                for i, lines in enumerate(f):

                    # Add a path of the file
                    metadata_dict['path'] = file

                    # Find the sample frequency
                    if i == 0:
                        fs = lines.split(' ')[2].strip()
                        metadata_dict['fs'] = int(fs)
                    
                    # Find the age information
                    if lines.startswith('#Age'):
                        age = lines.split(': ')[1].strip()
                        if age == 'NaN':
                            metadata_dict['age'] = -1
                        else:
                            metadata_dict['age'] = int(age)
                    
                    # Find the gender information
                    if lines.startswith('#Sex'):
                        gender = lines.split(': ')[1].strip()
                        if gender == 'NaN':
                            metadata_dict['gender'] = 'Unknown'
                        else:
                            metadata_dict['gender'] = gender

                metadata_rows.append(metadata_dict)

    return metadata_rows

def gather_metadata(files, labels, column_names):
    ''' Gather metadata of the files. Metadata can be either in header files
    or csv files.

    :param files: List-of-lists of ECG files for which the metadata is gathered
                  Each list contains files from a spesific database
    :type files: list-of-lists
    :param labels: SNOMED CT Codes that will be included in the classification
    :type labels: list
    :param column_names: Columns names for the final csv files of the files
    :type column_names: list

    :return: Dataframe which rows correspond to ECG samples. 
    :rtype: pandas.DataFrame
    '''

    ecg_rows = []
    # Iterate over different databases
    for file_set in files:

        # Check if metadata is in header files: If yes, ECGs should have corresponding hea files in the same location
        if os.path.basename(file_set[0]).endswith('.mat') and os.path.exists(file_set[0].replace('mat', 'hea')):
            ecg_rows.append(read_headerfiles(labels, file_set, column_names))
        
        # If not in a header file, must be in a csv file
        else:
            
            # First find the csv file and then extract the metadata
            dir_name = os.path.dirname(file_set[0])
            metacsv = glob.glob(os.path.join(dir_name, '*.csv'))
            assert len(metacsv) == 1, 'Something wrong with the csv file: Either not found or found too many.'

            ecg_rows.append(read_metacsv(labels, file_set, column_names, metacsv[0]))
    
    # Flatten a list-of-dictionaries to convert one dictionary to dataframe
    ecg_rows = [dct for dict_lst in ecg_rows for dct in dict_lst]
    return pd.DataFrame(ecg_rows)

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
    for i, db in enumerate(db_names):

        # Absolute path where to read the data from
        db_path = os.path.join(data_directory, db)
        
        # Getting files from the path
        filenames = [lsdir(db_path)]
        
        # Putting all the ECGs into a dataframe: first, create an empty one
        columns_names = ['path', 'age', 'gender', 'fs'] + labels

        # Excluding patients with diagnoses which are not included as labels for classification
        # Filling the empty dataframe
        ecg_df = gather_metadata(filenames, labels, columns_names)
        
        # Saving database-wise splitted data into csvs
        ecg_df.to_csv(os.path.join(save_directory, '{}.csv'.format(db)), sep=',', index=False)
        
        print('Created csv of the database {}!'.format(db))
        print('- Total of {} rows (excluded {} files as no wanted labels in them)'.format(len(ecg_df), len(filenames[0])-len(ecg_df)))

        # We also need to combine multiple databases as one csv file.
        # Let's think the already made one as the final test set and all the other
        # as training data, which are divided further into training and validation sets.
        train_data = db_names[:i] + db_names[i+1:]

        # We could think that only one database is leaved as validation set so
        # true training set is in size of len(train_data) - 1
        for combs in combinations(train_data, len(train_data)-1):
            train_csv_name = os.path.join(save_directory, '_'.join(sorted(combs, key=str.lower)) + '.csv')

            # If doesn't already exist, create the combined csv file
            if not os.path.exists(train_csv_name):
                filenames = [lsdir(os.path.join(data_directory, db_path)) for db_path in combs]
                ecg_df = gather_metadata(filenames, labels, columns_names)
                ecg_df.to_csv(train_csv_name, sep=',', index=False)

                print('Combined csv file ´{}´ created'.format(os.path.basename(train_csv_name)))
        
        print('-'*20)

        
def stratified_csvs(data_directory, save_directory, labels, train_test_splits):
    ''' Creating stratified data splits and saving them in csvs

    :param data_directory: The location of the data files
    :type data_directory: str
    :param save_directory: The location where to save the csvs to
    :type save_directory: str
    :param labels: Labels in SNOMED CT Codes
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

        # All the training files as a list of lists: can be .mat or .h5 files
        train_files = [lsdir(os.path.join(data_directory, db_path)) for db_path in train_data]

        # The train_files variable should be a list-of-lists, but if not, convert it
        train_files = train_files if len(train_files) > 1 else [train_files]

        # Putting all the ECGs into a dataframe: first, create an empty one
        column_names = ['path', 'age', 'gender', 'fs'] + labels
        
        # Gather metadata from the ECGs that are included in the classification
        ecg_df = gather_metadata(train_files, labels, column_names)
        
        # Training labels - the last columns of the dataframe
        train_labels = ecg_df.loc[:, (ecg_df.sum(axis=0) != 0)].iloc[:, 4:].values
        
        # Stratifying the data
        train_sets, val_sets = stratified_shuffle_split(ecg_df, train_labels, n_split)
        
        # Saving the stratified training and validation splits to csv files
        for i in range(n_split):
            train_sets[i].to_csv(os.path.join(save_directory, '{}.csv'.format('train_' + split + '_' + str(i+1))), sep=',', index=False)
            val_sets[i].to_csv(os.path.join(save_directory, '{}.csv'.format('val_' + split + '_' + str(i+1))), sep=',', index=False)

            if i == 0: # Save the test data only once
                test_data = data['test']
                test_path = os.path.join(data_directory, test_data)
                
                # Getting every test data file
                test_files = [lsdir(test_path)]

                # Gather the metadata for for the test data too
                test_set = gather_metadata(test_files, labels, column_names)
                test_set.to_csv(os.path.join(save_directory, '{}.csv'.format('test_' + split)), sep=',', index=False)
        
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
    data_dir = 'preprocessed_smoke_data'
    
    # ----- WHERE TO SAVE THE CSV FILES - give a name for the new directory
    # Note that the root for this is the 'data/split_csv/' directory
    csv_dir = 'stratified_smoke'

    # ----- LABELS TO USE IN SNOMED CT CODES: THESE ARE USED FOR CLASSIFICATION 
    labels = ['426783006', '426177001', '427084000', '164890007', '164889003', '427393009']
   
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
                'train': ['G12EC', 'Shandong', 'PTB_PTBXL', 'ChapmanShaoxing_Ningbo'],
                'test': 'CPSC_CPSC-Extra'
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
