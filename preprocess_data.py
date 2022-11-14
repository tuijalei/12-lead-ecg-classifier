import os
import numpy as np
from src.dataloader.transforms import Linear_interpolation, BandPassFilter
from distutils.dir_util import copy_tree
from scipy.io import loadmat, savemat

'''
With this script, the ECG data can be preprocessed before training.

First, the directory tree of the original data location is copied into a different directory 
so we keep the original data as it is and we can use created directories for data splitting, 
training and testing.

Second, all the ECGs and header files are paired and loaded directory by directory. The header 
files are needed for sample frequency as different frequencies are used in the data.

Third, the preprocessing is performed. Transforms are loaded from /src/dataloader/transforms.
    - BandPassFilter: filters out certain frequencies that lie within a particular band or 
                      range of frequencies
    - Linear_interpolation : resamples the ECG using linear interpolation
                      
Lastly, ECGs are saved. They need to be saved in mat format using dictionary with the key 'val'!
The original versions are deleted and only preprocessed ECGs are left in the new directory.

You are welcome to change the attributes `from_directory` and `new_directory` as you wish.

    from_directory      Where to load the original (not preprocessed) data
    new_directory       Where to save the preprocessed data
'''

# Original data location
from_directory = os.path.join(os.getcwd(), 'data', 'physionet_data_smoke')

# New location for preprocessed data
new_directory = os.path.join(os.getcwd(), 'data', 'physionet_preprocessed_smoke')
if not os.path.exists(new_directory):
    os.makedirs(new_directory)

print('Copying the directory tree...')
# Copy the directory tree
copy_tree(from_directory, new_directory)

# Subdirectories of the copied directory
directories = [os.path.join(new_directory, dir_tmp) for dir_tmp in os.listdir(new_directory) if not dir_tmp.startswith('.')]


mat_suffix = '.mat'
hea_suffix = '.hea'
# Iterate over directories and preprocess ECGs
for dire in directories:
    print('Opening {}...'.format(os.path.basename(dire)))
    
    # Get all files - each recording consists of a MatLab file and a headerfile
    mat_files = sorted([os.path.join(dire, file) for file in os.listdir(dire) if file.endswith(mat_suffix) and not 'preprocessed' in file])
    hea_files = sorted([os.path.join(dire, file) for file in os.listdir(dire) if file.endswith(hea_suffix) and not 'preprocessed' in file])

    # There should be as many mat files as there are header files
    assert(len(mat_files) == len(hea_files))
    print('Found total of {} mat files and {} hea files'.format(len(mat_files), len(hea_files)))

    print('Making pairs of hea and mat files...')
    # Similarly names files should be paired
    mat_hea_pairs = []
    for mat in mat_files:
        mat_name = os.path.basename(mat).split('.')[0]

        for hea in hea_files:
            hea_name = os.path.basename(hea).split('.')[0]
            if hea_name == mat_name:
                mat_hea_pairs.append((mat, hea))
                break # If file found, no need to continue

            
    print('Preprocessing {} ECGs...'.format(len(mat_hea_pairs)))
    # Iterate over mat and hea files and perform preprocessing
    for i, (mat, hea) in enumerate(mat_hea_pairs):

        # Get fs from header file
        with open(hea, 'r') as f:
            ecg_fs = int(f.readlines()[0].split(' ')[2])

        # Load ECG
        ecg = loadmat(mat)
        ecg = np.asarray(ecg['val'], dtype=np.float64)


        # ------------------------------
        # --- PREPROCESS TRANSFORMS ----

        # - BandPass filter 
        bpf = BandPassFilter(fs = ecg_fs)
        ecg = bpf(ecg)             
       
        # - Linear interpolation
        linear_interp = Linear_interpolation(fs_new = 250, fs_old = ecg_fs)
        ecg = linear_interp(ecg)

        # ------------------------------
        # ------------------------------

        # Since no need for the original ECG to exists in the 'preprocessed' directory
        # let's delete it
        os.remove(mat)
        
        # Save preprocessed ECG using dictionary with the key 'val'
        ecg_dict = {'val': ecg}
        ecg_name = os.path.splitext(mat)[0] + '_preprocessed.mat'
        savemat(ecg_name, ecg_dict)
         
        # We need a header file to have the same name as ECG so let's change it
        hea_name = os.path.splitext(hea)[0] + '_preprocessed.hea'
        os.rename(hea, hea_name)

        if i % 1000 == 0:
            print('{}/{} ECGs preprocessed'.format(i+1, len(mat_hea_pairs)))
    
    print('-'*20)

print('Done.')