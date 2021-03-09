import numpy as np
import scipy.io
import os
from Utilities.Saver import create_directory

directory = '/home/martint/Documents/MATLAB/tropex/Data/Exp11/DEBUGGING/10'
mat_directory = os.path.join(directory, 'chosen_mat')
np_directory = os.path.join(directory, 'chosen_arrays')
create_directory(np_directory)
for file_name in os.listdir(mat_directory):
    if file_name.split('.')[-1] == 'mat':
        old_path = os.path.join(mat_directory, file_name)
        new_path = os.path.join(np_directory, file_name)
        mat = scipy.io.loadmat(old_path)
        correlations = mat['chosen_correlations']
        np.save(new_path, correlations)
