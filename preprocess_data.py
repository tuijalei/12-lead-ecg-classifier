import os, sys, re, h5py, shutil
import pandas as pd
from src.dataloader.transforms import *
from src.dataloader.dataset_utils import load_data
from scipy.io import savemat

'''
With this script, the ECG data can be preprocessed before training. Supported ECG formats
are MATLAB v4 and h5, and metadata should be either in a header file or in a csv file.

You can add your own transforms inside the "PREPROCESS TRANSFORMS" block. 
By default the following transforms are used:
    - BandPassFilter: filters out certain frequencies that lie within a particular band or 
                      range of frequencies
    - Spline_interpolation: resamples the ECG using cubic spline interpolation
                      
The following attributes should be considered: 
    from_directory      Where to load the original (not preprocessed) data from
    new_directory       Where to save the preprocessed data

'''

# Original data location
from_directory = os.path.join(os.getcwd(), 'data', 'smoke_data')
assert os.path.exists(from_directory), 'The data directory doesn´t exist.'

# New location for preprocessed data
new_directory = os.path.join(os.getcwd(), 'data', 'preprocessed_smoke_data')

if not os.path.exists(new_directory):
    os.makedirs(new_directory)

print("Gather all the filenames from the given directory into a dictionary...") 

# Initialize a dictionary with keys that are directory names in the given directory
# If given one directory that includes files itself, have only this as a key
more_than_one = len(next(os.walk(from_directory))[1]) > 0
if more_than_one:
    files = {}
    # Also, create the subdirectories
    for dname in os.listdir(from_directory):
        prev_d = os.path.join(from_directory, dname)
        new_d = os.path.join(new_directory, dname)

        if os.path.isdir(prev_d) and not os.path.exists(new_d):
            os.makedirs(new_d)

        files[dname] = None
else:
    files = {os.path.basename(from_directory): None}

# Gather the filenames into the dictionary
for d in files.keys():
    d_path = os.path.join(from_directory, d) if more_than_one else from_directory
    filenames_tmp = [file for file in os.listdir(d_path)]
    files[d] = filenames_tmp

# File formats that are supported
ecg_suffix = ['h5', 'mat'] # no dot in these!
meta_suffix = ['csv', 'hea']

# Iterate over directories and preprocess ECGs
for d, filenames in files.items():
    print('Opening {}...'.format(os.path.basename(d)))
    
    # Absolute paths for the "old" and the new directories
    prev_path = os.path.join(from_directory, d) if more_than_one else from_directory
    new_path = os.path.join(new_directory, d) if more_than_one else new_directory

    # Get the absolute paths for ecgs and metadata
    ecg_files = sorted([os.path.join(prev_path, file) for file in filenames if re.search('\w+$', file)[0] in ecg_suffix])
    meta_files = sorted([os.path.join(prev_path, file) for file in filenames if re.search('\w+$', file)[0] in meta_suffix])
    assert len(ecg_files) > 0 and len(meta_files) > 0, 'If there are ecg files, there should be metadata too. Check if metadata found in the same location than ECGs!'

    # If the metadata is in a csv file, needs to be loaded only once
    if meta_files[0].endswith('csv'):
        assert len(meta_files) == 1, 'There should be only one csv file found from which metadata is read!'
        meta_df = pd.read_csv(meta_files[0])

    print('Preprocessing {} ECGs...'.format(len(ecg_files)))
    # Iterate over ecg recordings and preprocess them
    for i, ecg_name in enumerate(ecg_files):

        # Sample frequency is either in a csv file or in a hea file
        if meta_files[0].endswith('.hea'):
            # Doublecheck the naming of hea and mat files as they should match
            assert re.search('^\w+', os.path.basename(ecg_name))[0] == re.search('^\w+', os.path.basename(meta_files[i]))[0], 'Hea and mat files should have similar names!'

            with open(meta_files[i], 'r') as f:
                hea_file_lines = f.readlines()

            ecg_fs = int(hea_file_lines[0].split(' ')[2])

        else:
            # Double check that we have the metadata of the spesific ECG samples
            # i.e. it needs to be found in the ECG_ID column
            if os.path.basename(ecg_name) in meta_df['ECG_ID'].tolist():
                row_idx = meta_df.index[meta_df['ECG_ID'] == os.path.basename(ecg_name)]
                ecg_fs = int(meta_df.loc[row_idx, 'fs'])
            else: # If ECG not found from metadata, skip it
                continue

        # Load ECG
        ecg = load_data(ecg_name)

        # ------------------------------
        # --- PREPROCESS TRANSFORMS ----
        new_fs = 250
        
        # - BandPass filter 
        bpf = BandPassFilter(fs = ecg_fs)
        ecg = bpf(ecg)
        
        # - Spline interpolation
        si = Spline_interpolation(fs_new = new_fs, fs_old = ecg_fs)
        ecg = si(ecg)
        
        # ------------------------------
        # ------------------------------

        prev_name, suffix = os.path.splitext(os.path.basename(ecg_name))
        new_name = os.path.join(new_path, prev_name + '_preprocessed' + suffix)

        # If ECG is a .mat file, use the savemat function
        if ecg_name.endswith('.mat'):
            
            # Using dictionary with the 'val' key
            ecg_dict = {'val': ecg}
            savemat(new_name, ecg_dict)

            # We need the header file in the same location and 
            # to have a similar name so let's create one
            prev_name, suffix = os.path.splitext(os.path.basename(meta_files[i]))
            new_hea = os.path.join(new_path, prev_name + '_preprocessed' + suffix)

            # Doublecheck the names to be sure that same hea file is copied
            assert re.search('\D+\d+', os.path.basename(meta_files[i]))[0] == re.search('\D+\d+', os.path.basename(new_hea))[0], 'Hea files should have similar names except `_preprocessed` part!'
            shutil.copy(meta_files[i], new_hea)

            # Update also the hea file
            # First, replace the previous fs with the new one
            splitted_line = hea_file_lines[0].split(' ')
            splitted_line[2] = str(new_fs)
            new_line = ' '.join(splitted_line)
            hea_file_lines[0] = new_line

            # Rewrite the hea file with new sample frequency
            with open(new_hea, 'w') as f:
                f.write(''.join(hea_file_lines))

        else:
            # H5 files had a key named ´ecg´ to where to store the preprocessed ECG
            with h5py.File(new_name, 'w') as f:
                f['ecg'] = ecg
        
        if i % 1000 == 0:
            print('{:^8}/{:^8} ECGs preprocessed'.format(i+1, len(ecg_files)))
    
    print('-'*20)

    # Lastly, update the csv file of the metadata
    # Note, the names needs to be updated in the ECG_ID column
    #       and the sample frequency in the fs column
    if meta_files[0].endswith('csv'):
        new_csv = meta_df.copy()
        new_names = []

        for name in ecg_files:
            full_name = os.path.basename(name)

            if full_name in meta_df['ECG_ID'].tolist():
                prev_name, suffix = os.path.splitext(full_name)
                new_names.append(prev_name + '_preprocessed' + suffix)

        new_csv['ECG_ID'] = new_names
        new_csv['fs'] = new_fs

        new_csv.to_csv(os.path.join(new_path, os.path.basename(meta_files[0])), index=None, sep=',')

print('Done.')