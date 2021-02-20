import os
from Utilities.Data_Loader import load_data
from Utilities.Parser import parse_arguments
from Utilities.Network_Loader import load_network

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
arg = parse_arguments()

def save_training_accuracies(arg):
    x_train, y_train = load_data(arg, data_type='training')
    for epoch_number in range(200):
        network = load_network(arg, epoch_number)


a = 5