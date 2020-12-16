import numpy as np
import os
from Utilities.Tropical_Helper_Functions import get_function_path, get_current_data
from Utilities.Data_Loader import generate_adversarial_data, load_data, prepare_data_for_tropical_function
from Utilities.Network_Loader import load_network
from Utilities.Parser import parse_arguments
from Utilities.Saver import create_directory, get_saving_directory


def compute_accuracy(network, data_points, labels):
    return np.sum(np.argmax(network.predict(data_points), axis=1) == labels) / data_points.shape[0]


def compute_agreement(array1, array2):
    return np.sum(array1 == array2)/array1.size

arg = parse_arguments()
for i in range(5):
    arg.network_number = str(i)
    network = load_network(arg, None)
    x_test, y_test = load_data(arg, arg.data_type)
    network_labels = np.argmax(network.predict(x_test), axis=1)
    print(arg.data_type.capitalize() + ' accuracy, network number ' + str(i) + ':')
    print(np.sum(network_labels == y_test)/y_test.size)


save_dir = get_saving_directory(arg)
network = load_network(arg, None)
x_train, y_train, x_test, y_test = load_data(arg.data_set)
training_labels = np.argmax(y_train, axis=1)
test_labels = np.argmax(y_test, axis=1)
no_labels = y_train.shape[1]

x_train_adv = generate_adversarial_data(network, x_train, method='momentum_iterative')
x_test_adv = generate_adversarial_data(network, x_test, method='momentum_iterative')
train_accuracy = compute_accuracy(network, x_train, training_labels)
test_accuracy = compute_accuracy(network, x_test, test_labels)
train_adv_accuracy = compute_accuracy(network, x_train_adv, training_labels)
test_adv_accuracy = compute_accuracy(network, x_test_adv, test_labels)

last_layer_index = len(network.layers) - 2
training_directory = create_directory(save_dir, 'Training')
function_path = get_function_path(arg, last_layer_index, training_directory)

x_train_adv_tropical = prepare_data_for_tropical_function(x_train_adv)
out = [None] * no_labels
for j in range(no_labels):
    pos_terms_j = np.load(function_path + 'pos_label_' + str(j) + '.npy')
    out[j] = np.max(np.dot(pos_terms_j, x_train_adv_tropical), axis=0)
tropical_adv_labels = np.argmax(out, axis=0)
tropical_adv_accuracy = np.sum(tropical_adv_labels == training_labels)/y_train.shape[0]
equal_after_adv = compute_agreement()
a = 5


