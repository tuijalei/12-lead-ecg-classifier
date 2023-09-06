import os, re, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def find_headerfiles(input_dir):
    ''' Find the headerfiles of the Physionet Challenge 2021 data and store the whole paths    
    '''
    
    header_files = [] # paths of the header files
    for root, _, files in os.walk(input_dir):
        
        for file in sorted(files):
            if not file.lower().startswith('.') and file.lower().endswith('hea'):
                g = os.path.join(root, file)
                header_files.append(g)

    return header_files
    

def physionet_metadata(files):
    ''' Extract the metadata from header files of the Physionet Challenge 2021 dataset.
    For mapping the labels, only the diagnosis label are needed.
    '''
    metadata_rows = []

    # Iterate over files and find the diagnostic labels
    for filename in files:
        
        with open(filename, 'r') as f:
            f_name = re.search("\w*.\d\w.hea", filename).group()
            dxs = []
            
            for line in f:
                
                # Get needed information
                if line.startswith('#Dx'):
                    dxs = line.split(': ')[1].split(',')
                    dxs = [dx.strip() for dx in dxs]

            info = {'SNOMEDCTCode': dxs,
                    'file': f_name}
            
            metadata_rows.append(info)
    
    return pd.DataFrame(metadata_rows)

def merge_labels(df):
    ''' Merge one label to another, i.e., map a new diagnostic code for the patient.'''

    prolonged_pr_snomed = '164947007'
    first_degree_hb_snomed = '270492004'

    # Let's find all prolonged pr intervals
    pr_idx = [i for i, row in df.iterrows() if prolonged_pr_snomed in list(map(str, row['SNOMEDCTCode']))]

    # How many of these have both diagnosis?
    both_idx = [i for i, row in df.loc[pr_idx, :].iterrows() if first_degree_hb_snomed in list(map(str, row['SNOMEDCTCode']))]

    # Find patients that lack wanted diagnosis
    merge_idx = [dx for dx in pr_idx if not dx in both_idx]

    # Append "1st degree heart block" to the patients that doesn't have it yet
    for dx in df.loc[merge_idx, 'SNOMEDCTCode']:
        dx.append(str(first_degree_hb_snomed))

    return df

def feature_matrix(data, labels):
    ''' Create a feature matrix dataframe where the columns are SNOMED CT Codes and 
    rows correspond to each ECG. One hot encode diagnostic labels with the values 0 and 1.
    Include defined labels only so drop the ECGs (rows) which doesn't have any included labels.
    Use filenames as indeces for the created feature matrix to keep track of files left in the dataframe.
    '''

    # Create a feature matrix in size <num of ecgs (rows)> X <num of labels (cols)>
    f_matrix =  np.zeros((data.shape[0], len(labels)), dtype=int)

    # Fill up the feature matrix: Map the found SNOMED CT Codes with the value of 1
    for i, dx in enumerate(data['SNOMEDCTCode']):

        # If "normal ecg" mapped with -1 is found, no need to do anything
        # as later we try to predict the SR label for them
        if '-1' in str(dx):
            continue
        
        # Find the diagnostic codes
        codes = re.findall('\w+', str(dx))
        codes = list(set(codes).intersection(set(labels)))

        # Mark only the diagnostic codes that are found from the labels variable
        if bool(codes):

            if len(codes) > 1: # multiple diagnostic codes found
                for d in codes:
                    d_index = labels.index(d)
                    f_matrix[i][d_index] = 1

            else: # only one found
                d_index = labels.index(codes[0])
                f_matrix[i][d_index] = 1

    # Create a feature dataframe and drop rows that contain only zeros
    f_matrix = pd.DataFrame(f_matrix, columns=labels)

    # Use filenames as indexes
    filename_cols = set(['file', 'ECG_ID'])
    f_matrix.index = data.loc[f_matrix.index, set(data.columns).intersection(filename_cols).pop()]

    return f_matrix

def label_mapping(metadata, label_map, from_code, to_code):
    ''' The function is made to convert AHA-codes in the Shandong data set to SNOMED CT Codes. 
    The dataset by Hui et al 2022 can be found here: https://www.nature.com/articles/s41597-022-01403-5#code-availability
    
    The AHA coded diagnoses have the following structure in the metadata file:

        60+310;50+346;147

    meaning that diagnostic statements include primary statement codes (the first numbers, e.g. 60 and 50) and
    modifiers (after the + mark, e.g. 310 and 346), and the statements are separated from each other using the ; mark.
    '''

    # Iterate over the metadata file and add a new diagnosis code if possible
    for index, values in metadata.iterrows():

        code_orig = str(values[from_code])

        # If there is a + mark in diagnosis, there are modifiers
        if '+' in code_orig:
            codes = code_orig.split(';')
        
        else:
            codes = re.findall('\d+', code_orig) # might be one or multiple

        # Find normal ecgs and map them with "-1"
        if '1' in codes:
            metadata.loc[index, to_code] = '-1'

        # Check if any code in metadata found from the mapping csv file
        # If yes, find corresponding SNOMED CT Codes
        if any(c for c in codes if str(c) in label_map[from_code].values):
            found_codes = [c for c in codes if str(c) in label_map[from_code].values]
        
            # Gather codes
            snomed = []
            for dx in found_codes:
                snomed_tmp = label_map.loc[label_map[from_code] == dx, to_code].tolist()

                # There might be multiple SNOMED CT Codes for one AHA statement
                if len(snomed_tmp) == 1:
                    snomed.append(snomed_tmp[0])
                else:
                    for s in snomed_tmp:
                        snomed.append(s)
                
            # If codes, convert into a string of codes and store using row index
            if bool(snomed):
                metadata.loc[index, to_code] = ','.join(list(map(str, set(snomed))))

    return metadata



if __name__ == '__main__':

    '''
    The script has been written to enable the use of the Physionet Challenge 2021 data and the Shandong Provincial Hospital (SPH) data
    by Hui et al. 2022 (https://www.nature.com/articles/s41597-022-01403-5#code-availability) in this repository. The two data
    are using different diagnostic codes so label mapping is needed to convert AHA codes (SPH) to SNOMED CT Codes (Physionet).
    Modeling in this repository is based on the SNOMED CT Codes so all other different diagnostic codes needs to be converted 
    into them. The script updates the original dataframe with a new column, e.g. 'SNOMEDCTCode'.

    The header files and csv file are the two supported types for metadata. The metadata files needs to be in the same directory
    where the ECG data exists. If another csv file is stored, make sure to keep only one csv file in the data directory so that 
    the script creating csv files of data splits knows where to load the codes from. I.e., delete or move the ones that are not needed.

    Note also that the columns should be similarly named in the metadata file and the mapping file. E.g. To find the AHA codes 
    used in the SPH data, the "AHA_Code" named column is looked for, and the similarly named column from the mapping file
    is looked for to convert those AHA codes to SNOMED CT Codes.

    The script can also perform imputation of the sinus rhythm label to the new data if it's missing. Sinus rhythm label is used
    by the Physionet Challenge 2021 score, so if the metric is wanted to be computed, the imputation should be considered.

    '''

    # ---- CSV FILE OF THE METADATA
    csv_path = os.path.join(os.getcwd(), 'data', 'smoke_data', 'SPH', 'metadata.csv')
    csv_save_path = os.path.join(os.getcwd(), 'data', 'smoke_data', 'SPH', 'updated_metadata_SPH.csv')

    # ---- CSV FILE OF LABEL MAPPING
    map_path = os.path.join(os.getcwd(), 'data', 'AHA_SNOMED_mapping.csv')

    # --- FROM WHICH AND TO WHICH TO CONVERT
    #     corresponds to the columns from where the codes are read from metadata and mapping file
    from_code = 'AHA_Code'
    to_code = 'SNOMEDCTCode'

    # --- The imputation choice: Will the sinus rhythm label be needed within the new data set?
    imputation = True

    # --- Diagnostic labels in SNOMED CT Codes wanted to be included in the imputation
    sinus_rhythm = '426783006'
    labels = [sinus_rhythm, '426177001', '164934002', '427393009', '713426002', '427084000', '59118001', '164889003', '59931005', \
              '47665007', '445118002', '39732003', '164890007', '164909002', '270492004', '251146004', '284470004']
    
    # --- Which directory to use to train the Logistic Regression model for the imputation
    input_dir = os.path.join(os.getcwd(), 'data', 'physio_sph')
    
    # -------------------------------------------------------------------------------

    # Read the csv file containing metadata
    sph_metadata = pd.read_csv(csv_path)

    # Add sample frequencies (needed when creating csv files for modeling part)
    sph_metadata['fs'] = 500

    # Create an empty column to which to gather the SNOMED CT Codes
    sph_metadata[to_code] = np.nan

    # Make sure the ECGs are named in the csv file as they are named in the folder 
    # The csv file should be in the same folder as the ECGs are!
    ecg_names = sorted([file for file in os.listdir(os.path.dirname(csv_path)) if not file.endswith('.csv')])
    if not ecg_names == sph_metadata['ECG_ID'].tolist():
        sph_metadata['ECG_ID'] = ecg_names

    # Get label mapping
    label_map = pd.read_csv(map_path, sep=',')

    # Found the corresponding labels from another diagnosis coding system
    print('Converting AHA codes to SNOMED CT ones...')
    sph_metadata = label_mapping(sph_metadata, label_map, from_code, to_code)
    
    # Drop the rows that doesn't contain neither SNOMED CT Code or '-1' (normal ecg label)
    sph_metadata = sph_metadata.dropna()
    sph_metadata = sph_metadata.reset_index(drop=True)

    # Impute the sinus rhythm if necessary
    if bool(imputation):
        print('Performing imputation of the SR labels...')
        
        # Load the Physionet data
        print('Loading the Physionet Challenge 2021 data...')
        physionet_heas = find_headerfiles(input_dir)
        physionet_data = physionet_metadata(physionet_heas)

        # Merge "prolonged PR interval" to "1st degree HB"
        physionet_data = merge_labels(physionet_data)

        # Convert diagnoses into one hot encoding and drop the ECGs that don't include defined SNOMED CT Codes
        physio_feature_matrix = feature_matrix(physionet_data, labels)

        # Drop all rows that contain only 0 values
        physio_feature_matrix = physio_feature_matrix.loc[~(physio_feature_matrix==0).all(axis=1)]

        # Labels for SR imputation are the SR labels themselves and the other labels are the features
        physio_labels = physio_feature_matrix.loc[:, sinus_rhythm].values.tolist()
        physio_features = physio_feature_matrix.drop([sinus_rhythm], axis = 1)

        # =========== IMPUTATION OF SR LABELS ===========
        print('Fitting the Logistic Regression model with the Physionet metadata...')

        # Fit a logistic regression model
        logreg = LogisticRegression(C=0.01, max_iter=1000).fit(physio_features, physio_labels)

        # Predict SR labels for the SPH data
        # First, make the feature matrix out of the SPH metadata
        print('Predicting SR labels for the SPH data...')
        sph_feature_matrix = feature_matrix(sph_metadata, labels).drop([sinus_rhythm], axis = 1)
        sr_predictions = logreg.predict(sph_feature_matrix)

        # Store the predicted SR labels in the SPH metadata
        sph_metadata['SR'] = sr_predictions

        # Lastly, add SNOMED CT Code of the SR along the other SNOMED CT Codes to the SPH data if SR predicted
        print('Converting SR predictions into SNOMED CT Codes...')
        for index, values in sph_metadata.iterrows():

            if values['SR'] == 1:

                snomeds = str(values['SNOMEDCTCode'])
                
                if snomeds == '-1': # map normal ecgs
                    sph_metadata.loc[index, 'SNOMEDCTCode'] = sinus_rhythm
                    
                else: # map everything else
                    snomeds = snomeds + ',' + str(sinus_rhythm)
                    sph_metadata.loc[index, 'SNOMEDCTCode'] = snomeds

    # Save the updated csv file
    sph_metadata.to_csv(csv_save_path, index=False)
    print('Done.')
