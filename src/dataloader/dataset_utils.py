from scipy.io import loadmat
import numpy as np
import sys, h5py
from biosppy.signals.tools import filter_signal
from scipy.signal import decimate, resample

def load_data(case):
    ''' Load a MATLAB v4 file or a H5 file of an ECG recording
    '''

    if case.endswith('.mat'):
        x = loadmat(case)
        return np.asarray(x['val'], dtype=np.float64)
    else:
        with h5py.File(case) as f:
            x = f['ecg'][()]
        return np.asarray(x, dtype=np.float64)


def normalize(seq, smooth=1e-8):
    ''' Normalize each sequence between -1 and 1 '''
    return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1   

def apply_filter(signal, filter_bandwidth, fs=500):
        # Calculate filter order
        order = int(0.3 * fs)
        # Filter signal
        signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                     order=order, frequency=filter_bandwidth, 
                                     sampling_rate=fs)
        return signal

def standardize_sampling_rate(recording, ecg_fs, fs=500):
    ''' Standardize sampling rate ''' 
    if ecg_fs > fs:
        recording = decimate(recording, int(ecg_fs / fs))
    elif ecg_fs < fs:
        recording = resample(recording, int(recording.shape[-1] * (fs / ecg_fs)), axis=1)
    return recording     

def encode_metadata(age, gender):
    ''' Encode age and gender information

    :param age: Patient's age
    :type age: float
    :param gender: Parient's gender
    :type gender: str

    :return data: Array for representing patient's age and gender
                  if age: age/100
                  if female:      [0.83.   1.   0. ]
                  if male:        [0.83.   0.   1. ]
                  if Unknown/NaN: [0.83.   0.   0. ]
    :rtype: numpy.ndarray
    '''

    ag_data = np.zeros(3,)
    if age >= 0:
        ag_data[0] = age / 100
      
    if gender == 'Female' or gender == 'female' or gender == 'F' or gender == 'f':
        ag_data[1] = 1
    elif gender == 'Male' or gender == 'male' or gender == 'M' or gender == 'm':
        ag_data[2] = 1

    return ag_data