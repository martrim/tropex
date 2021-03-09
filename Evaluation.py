import numpy as np
from Utilities.Tropical_Helper_Functions import compute_maximal_difference, evaluate_tropical_function, get_current_data,\
    get_last_layer_index, get_tropical_function_directory, get_tropical_test_labels, load_tropical_function, \
    get_no_labels, get_epoch_numbers, get_max_data_group_size, evaluate_network_on_subgrouped_data, get_folder_name, \
    get_batch_data
from Utilities.Custom_Settings import apply_resnet_settings, configure_gpu
from Utilities.Logger import *
from Utilities.Network_Loader import load_network
from Utilities.Parser import parse_arguments
from Utilities.Saver import get_saving_directory

start_time = print_start()

# To raise an exception on runtime warnings, used for debugging.
np.seterr(all='raise')

# Load the arguments
arg = parse_arguments()
if arg.network_type_coarse == 'ResNet':
    arg = apply_resnet_settings(arg)

# Configure the GPU for Tensorflow
configure_gpu(arg)


# COMPUTATION OF ACCURACY OF TROPICAL FUNCTION
def evaluate_x_test(arg, layer_idx):
    grouped_data, true_labels, network_labels = get_batch_data(arg, network)
    if layer_idx == 'all':
        lower_idx = 0
        upper_idx = last_layer_index + 1
    else:
        lower_idx = layer_idx
        upper_idx = layer_idx + 1
    accuracies = []
    for i in range(lower_idx, upper_idx):
        tropical_test_labels = get_tropical_test_labels(arg, network, grouped_data, i, epoch_number)
        if tropical_test_labels is not None:
            tropical_accuracy = sum(tropical_test_labels == true_labels) / len(true_labels)
            tropical_network_agreement = sum(tropical_test_labels == network_labels) / len(network_labels)
            logger.info(
                'Agreement of the tropical function and the neural network on the test set: ' + str(
                    tropical_network_agreement))
            logger.info('Agreement of the tropical function and the true labels on the test set: ' + str(
                tropical_accuracy))
            accuracies.append(np.array([str(tropical_network_agreement), str(tropical_accuracy)]))
    accuracies = np.array(accuracies)
    save_dir = get_saving_directory(arg)
    np.savetxt(os.path.join(save_dir, "accuracies.csv"), accuracies, delimiter=",", fmt='%s')


def evaluate_x_train(arg, layer_idx):
    def compute_max_error(layer_idx):
        folder_name = get_folder_name(network, layer_idx)
        function_path = get_tropical_function_directory(arg, folder_name, 'training', epoch_number)
        current_layer_name = function_path.split('/')[-2]
        logger.info('After merging with layer ' + current_layer_name + ':')
        current_data = get_current_data(network, grouped_data, layer_idx)
        pos_terms, true_labels, network_labels = load_tropical_function(arg, folder_name, arg.data_type, epoch_number, sign='pos')
        terms, _, _ = load_tropical_function(arg, folder_name, arg.data_type, epoch_number, sign='neg')
        pos_result, neg_result = evaluate_tropical_function(current_data, network_labels, pos_terms, terms)
        tropical_labels = np.argmax(pos_result, axis=0)
        results = str(sum(tropical_labels == network_labels) / len(network_labels))
        logger.info(
            'Agreement of the tropical function and the neural network on the training set: ' + str(results))
        tropical_labels = np.expand_dims(tropical_labels, axis=0)
        max_pos_result = np.take_along_axis(pos_result, tropical_labels, axis=0)
        max_neg_result = np.take_along_axis(neg_result, tropical_labels, axis=0)
        max_result = max_pos_result - max_neg_result
        logger.info('Maximal Error: ' + str(compute_maximal_difference(x_train_predicted, max_result)))

    grouped_data, true_labels, network_labels = get_batch_data(arg, network)
    x_train_predicted = evaluate_network_on_subgrouped_data(network, grouped_data)

    if layer_idx == 'all':
        for i in range(last_layer_index + 1):
            compute_max_error(i)
    else:
        compute_max_error(layer_idx)


logger = get_logger(arg)
epoch_numbers = get_epoch_numbers(arg)

for epoch_number in epoch_numbers:
    print('Epoch number: ' + str(epoch_number))
    logger.info('Epoch number: ' + str(epoch_number))
    network = load_network(arg, epoch_number)
    last_layer_index = get_last_layer_index(network)
    no_labels = get_no_labels(network)

    if arg.data_type == 'training':
        evaluate_x_train(arg, layer_idx=0)
    elif arg.data_type == 'test':
        evaluate_x_test(arg, layer_idx=0)

print_end(start_time)