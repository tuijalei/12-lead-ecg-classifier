from scipy.io import loadmat
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

def load_data(case):
    ''' Load a MATLAB v4 file of an ECG recording
    '''
    x = loadmat(case)
    return np.asarray(x['val'], dtype=np.float64)
   

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
      
    if gender == 'Female' or gender == 'female':
        ag_data[1] = 1
    elif gender == 'Male' or gender == 'male':
        ag_data[2] = 1

    return ag_data