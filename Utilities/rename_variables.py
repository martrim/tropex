import os
from scipy.io import loadmat, savemat

file_directory = '/media/martint/10A4A32CA4A312F0/'
file_name = 'FCN_MNIST_retrain_angles_correlations_distances_epoch.mat'
file_path = os.path.join(file_directory, file_name)
Dict = loadmat(file_path)

Dict.pop('__header__')
Dict.pop('__version__')
Dict.pop('__globals__')

Dict['MNIST_angles'] = Dict.pop('-_angles')
Dict['MNIST_correlations'] = Dict.pop('-_correlations')
Dict['MNIST_distances'] = Dict.pop('-_distances')

savemat(file_path, Dict)
a = 5