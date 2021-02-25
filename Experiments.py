from multiprocessing import Manager, Process
import pickle
import numpy as np
from functools import reduce
from scipy.io import savemat
from scipy.spatial.distance import cdist
from Utilities.Custom_Settings import apply_resnet_settings
from Utilities.Tropical_Helper_Functions import evaluate_tropical_function, get_current_data, get_grouped_data, \
    get_last_layer_index, get_no_labels, get_tropical_test_labels, load_tropical_function, shift_array, \
    get_tropical_function_directory, evaluate_batch_of_tropical_function, flatten_and_stack, \
    load_tropical_function_batch, \
    get_epoch_numbers, compute_1_1_euclidean_distances, compute_1_1_angles, get_associated_training_points, \
    get_max_data_group_size, reorder_terms, \
    get_activation_patterns, stack_list_with_subgroups, group_points, partition_according_to_correct_indices
from Utilities.Data_Loader import load_data
from Utilities.Logger import *
from Utilities.Network_Loader import load_network
from Utilities.Parser import parse_arguments
from Utilities.Saver import get_saving_directory

start_time = print_start()

# Load the arguments
arg = parse_arguments()
if arg.network_type_coarse == 'ResNet':
    arg = apply_resnet_settings(arg)

# Set available GPU
os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu


# EXPERIMENTS
def compute_no_cross_terms(arg):
    pos_out = [None] * no_labels
    neg_out = [None] * no_labels
    equal_indices = [None] * no_labels
    function_path = get_function_path(arg, last_layer_index, transformation_path)
    current_data = get_current_data(network, grouped_data, last_layer_index)
    for label_idx in range(no_labels):
        pos_terms_j = np.load(function_path + 'pos_label_' + str(label_idx) + '.npy')
        pos_values_j = np.dot(pos_terms_j, current_data)
        pos_indices_j = np.argmax(pos_values_j, axis=0)
        pos_out[label_idx] = pos_values_j[pos_indices_j, range(pos_values_j.shape[1])]
        neg_terms_j = np.load(function_path + 'neg_label_' + str(label_idx) + '.npy')
        neg_values_j = np.dot(neg_terms_j, current_data)
        neg_indices_j = np.argmax(neg_values_j, axis=0)
        neg_out[label_idx] = neg_values_j[neg_indices_j, range(neg_values_j.shape[1])]
        equal_indices[label_idx] = (pos_indices_j == neg_indices_j)
    equal_indices = np.array(equal_indices)
    tropical_test_labels = np.argmax(pos_out, axis=0)
    neg_test_labels = np.argmax(neg_out, axis=0)
    equal_labels = (tropical_test_labels == neg_test_labels)
    equal_index_and_label = np.logical_and(equal_indices, equal_labels)
    straight_terms = equal_index_and_label[tropical_test_labels, range(equal_index_and_label.shape[1])]
    cross_terms = np.logical_not(straight_terms)
    no_straight_terms = np.sum(straight_terms)
    no_cross_terms = np.sum(cross_terms)
    tropical_straight_labels = tropical_test_labels[straight_terms]
    true_straight_labels = true_labels[straight_terms]
    network_straight_labels = network_labels[straight_terms]
    results = str(sum(tropical_straight_labels == network_straight_labels) / len(network_straight_labels))
    logger.info(
        'On straight terms: Agreement of the tropical function and the neural network on the test set: '
        + str(results))
    results = str(sum(tropical_straight_labels == true_straight_labels) / len(true_straight_labels))
    logger.info(
        'On straight terms: Agreement of the tropical function and the true labels on the test set: '
        + str(results))
    if no_cross_terms > 0:
        tropical_cross_labels = tropical_test_labels[cross_terms]
        true_cross_labels = true_labels[cross_terms]
        network_cross_labels = network_labels[cross_terms]
        results = str(sum(tropical_cross_labels == network_cross_labels) / len(network_cross_labels))
        logger.info(
            'On cross terms: Agreement of the tropical function and the neural network on the test set: '
            + str(results))
        results = str(sum(tropical_cross_labels == true_cross_labels) / len(true_cross_labels))
        logger.info(
            'On cross terms: Agreement of the tropical function and the true labels on the test set: '
            + str(results))
    logger.info('Number of straight terms: ' + str(no_straight_terms))
    logger.info('Number of cross terms: ' + str(no_cross_terms))


def count_no_linear_regions():
    training_path = transformation_path + 'Training/'
    subdirectories = [x[0] for x in os.walk(training_path)]
    training_directory = list(filter(lambda x: 'Completed' in x, subdirectories))[0] + '/'
    testing_path = transformation_path + 'Test/'
    subdirectories = [x[0] for x in os.walk(testing_path)]
    test_directory = list(filter(lambda x: 'Completed' in x, subdirectories))[0] + '/'
    no_labels = get_no_labels(network)
    pos_max = [None] * no_labels
    for i in range(no_labels):
        pos_max[i] = np.load(training_directory + 'pos_label_' + str(i) + '.npy')
    pos_max = np.vstack(pos_max)
    neg_max = np.load(training_directory + 'neg_label_all.npy')
    pos_max_test = [None] * no_labels
    for i in range(no_labels):
        pos_max_test[i] = np.load(test_directory + 'pos_label_' + str(i) + '.npy')
    pos_max_test = np.vstack(pos_max_test)
    pos_max = np.vstack([pos_max, pos_max_test])
    neg_max = np.vstack([neg_max, np.load(test_directory + 'neg_label_all.npy')])
    diff = np.round(pos_max - neg_max, decimals=3)
    unique = np.unique(diff, axis=0)
    no_linear_regions = unique.shape[0]
    return no_linear_regions


def compare_test_and_training_images(arg, lower_true_label, idx):
    import matplotlib.pyplot as plt

    current_idx = idx
    x_train, y_train, x_test, y_test = load_data(arg.data_set)
    x_test_grouped = group_points(x_test, y_test)
    training_terms = [None] * no_labels
    training_path = get_function_path(arg, last_layer_index, transformation_path)
    for i in range(no_labels):
        training_terms[i] = np.load(training_path + 'pos_label_' + str(i) + '.npy')

    for true_label in range(lower_true_label, no_labels):
        plotted = False
        while not plotted:
            image = x_test_grouped[true_label][current_idx:current_idx + 1]
            image_network_label = np.argmax(network.predict(image), axis=1)[0]
            if not (true_label == image_network_label):
                current_idx = current_idx + 1
            else:
                reshaped_test_image = np.append(1, np.reshape(image, newshape=[-1]))
                outputs = [None] * no_labels
                for i in range(no_labels):
                    outputs[i] = np.max(np.dot(training_terms[i], reshaped_test_image), axis=0)
                image_tropical_label = np.argmax(outputs)
                training_idx = np.argmax(np.dot(training_terms[image_tropical_label], reshaped_test_image), axis=0)
                network_labels_train = np.argmax(network.predict(x_train), axis=1)
                x_train_grouped = group_points(x_train, network_labels_train)
                corresponding_training_image = x_train_grouped[image_tropical_label][training_idx:training_idx + 1]
                if (true_label != image_tropical_label):
                    fig2 = plt.figure()
                    im_train = plt.imshow(corresponding_training_image.squeeze())
                    plt.title('Corresponding Training Image')
                    plt.show()
                    plt.axis('off')

                    fig = plt.figure()
                    plt.title('Test Image')
                    im_test = plt.imshow(image.squeeze())
                    plt.show()
                    plt.axis('off')

                    plotted = True
                    current_idx = idx
                    plt.close('all')
                else:
                    current_idx = current_idx + 1


def val_accuracy_and_false_classification():
    val_data = np.vstack(grouped_data)
    val_data = np.reshape(val_data, newshape=[val_data.shape[0], -1])
    val_data = np.hstack([np.ones_like(val_data[:, 0:1]), val_data]).transpose()
    out = [None] * no_labels
    for j in range(no_labels):
        file_name = 'pos_label_' + str(j) + '.npy'
        pos_terms_j = np.load(transformation_path + file_name)
        results = np.dot(pos_terms_j, val_data)
        np.fill_diagonal(results[:, 5000 * j:5000 * (j + 1)], 0)
        out[j] = np.max(results, axis=0)
    validation_labels = np.argmax(out, axis=0)
    print(np.sum(validation_labels == network_labels) / 50000)
    print(np.sum(validation_labels == true_labels) / 50000)
    b = 5


def random_training_sampling():
    function_path = get_function_path(arg, last_layer_index, transformation_path)
    current_data = get_current_data(network, grouped_data, last_layer_index)
    no_runs = 5
    exp4_results = np.zeros([no_runs, 9 * 2])
    for i in range(1, 10):
        logger.info("Training Data Point Proportion: {}".format(i / 10))
        out = [None] * no_labels
        for k in range(no_runs):
            for j in range(no_labels):
                pos_terms_j = np.load(function_path + 'pos_label_' + str(j) + '.npy')
                no_training_indices = int(pos_terms_j.shape[0] * i / 10)
                training_indices = np.random.choice(pos_terms_j.shape[0], size=[no_training_indices],
                                                    replace=False)
                pos_terms_j = pos_terms_j[training_indices, :]
                out[j] = np.max(np.dot(pos_terms_j, current_data), axis=0)
            tropical_test_labels = np.argmax(out, axis=0)
            results = str(sum(tropical_test_labels == network_labels) / len(network_labels))
            exp4_results[k, (i - 1)] = results
            logger.info(
                'Agreement of the tropical function and the neural network on the test set: ' + str(results))
            results = str(sum(tropical_test_labels == true_labels) / len(true_labels))
            exp4_results[k, 9 + (i - 1)] = results
            logger.info('Agreement of the tropical function and the true labels on the test set: ' + str(results))
    np.savetxt(transformation_path + "exp4_results.csv", exp4_results, delimiter=",")


def compute_translation_invariance(shift_type):
    network_accuracy = sum(network_labels == true_labels) / len(true_labels)
    tropical_labels = evaluate_x_test(last_layer_index)
    tropical_accuracy = sum(tropical_labels == true_labels) / len(true_labels)

    stacked_data = np.vstack(grouped_data)
    exp5_network_remaining_accuracy = np.zeros([6, 6])
    exp5_tropical_remaining_accuracy = np.zeros([6, 6])
    exp5_network_same_label = np.zeros([6, 6])
    exp5_tropical_same_label = np.zeros([6, 6])
    shifts = [-5, -3, -1, 1, 3, 5]
    for i in range(6):
        for j in range(6):
            shifted_data = shift_array(stacked_data, shifts[i], shifts[j], shift_type)
            # Computing the accuracy of the network on the shifted data.
            network_labels_shifted = np.argmax(network.predict(shifted_data), axis=1)
            network_accuracy_shifted = sum(network_labels_shifted == true_labels) / len(true_labels)
            exp5_network_remaining_accuracy[i, j] = str(network_accuracy_shifted / network_accuracy)
            exp5_network_same_label[i, j] = str(np.sum(network_labels == network_labels_shifted) / len(network_labels))
            # Computing the accuracy of the tropical function on the shifted data.
            current_data = np.reshape(shifted_data, newshape=[shifted_data.shape[0], -1])
            current_data = np.hstack([np.ones_like(current_data[:, 0:1]), current_data])
            current_data = current_data.transpose()
            function_path = get_function_path(arg, last_layer_index, transformation_path)
            out = [None] * no_labels
            for k in range(no_labels):
                pos_terms_k = np.load(function_path + 'pos_label_' + str(k) + '.npy')
                out[k] = np.max(np.dot(pos_terms_k, current_data), axis=0)
            tropical_labels_shifted = np.argmax(out, axis=0)
            tropical_accuracy_shifted = sum(tropical_labels_shifted == true_labels) / len(true_labels)
            exp5_tropical_remaining_accuracy[i, j] = str(tropical_accuracy_shifted / tropical_accuracy)
            exp5_tropical_same_label[i, j] = str(
                np.sum(tropical_labels == tropical_labels_shifted) / len(tropical_labels))
            np.savetxt(transformation_path + "exp5_network_remaining_accuracy.csv", exp5_network_remaining_accuracy,
                       delimiter=",")
            np.savetxt(transformation_path + "exp5_tropical_remaining_accuracy.csv", exp5_tropical_remaining_accuracy,
                       delimiter=",")
            np.savetxt(transformation_path + "exp5_network_same_label.csv", exp5_network_same_label,
                       delimiter=",")
            np.savetxt(transformation_path + "exp5_tropical_same_label.csv", exp5_tropical_same_label,
                       delimiter=",")


def check_implications():
    grouped_data, true_labels, network_labels = get_grouped_data(arg, network, 'test', get_labels=True)
    tropical_test_labels = get_tropical_test_labels(arg, network, grouped_data, last_layer_index)

    network_true = (network_labels == true_labels)
    network_false = np.logical_not(network_true)
    tropical_true = (tropical_test_labels == true_labels)
    tropical_false = np.logical_not(tropical_true)
    network_tropical_same = (network_labels == tropical_test_labels)
    network_tropical_different = np.logical_not(network_tropical_same)
    network_tropical_false = np.logical_and(network_false, tropical_false)

    # network true => Tropical true?
    print('network true => Tropical true?')
    print(np.sum(tropical_true[network_true]) / np.sum(network_true))
    # network true => Tropical false?
    print('network true => Tropical false?')
    print(np.sum(tropical_false[network_true]) / np.sum(network_true))
    # network false => Tropical true?
    print('network false => Tropical true?')
    print(np.sum(tropical_true[network_false]) / np.sum(network_false))
    # network false => Tropical false?
    print('network false => Tropical false?')
    print(np.sum(tropical_false[network_false]) / np.sum(network_false))

    # Tropical true => network true?
    print('Tropical true => network true?')
    print(np.sum(network_true[tropical_true]) / np.sum(tropical_true))
    # Tropical true => network false?
    print('Tropical true => network false?')
    print(np.sum(network_false[tropical_true]) / np.sum(tropical_true))
    # Tropical false => network true?
    print('Tropical false => network true?')
    print(np.sum(network_true[tropical_false]) / np.sum(tropical_false))
    # Tropical false => network false?
    print('Tropical false => network false?')
    print(np.sum(network_false[tropical_false]) / np.sum(tropical_false))

    # network false => Same label?
    print('network false => Same label?')
    print(np.sum(network_tropical_same[network_false]) / np.sum(network_false))
    # network false => Different label?
    print('network false => Different label?')
    print(np.sum(network_tropical_different[network_false]) / np.sum(network_false))
    # Tropical false => Same label?
    print('Tropical false => Same label?')
    print(np.sum(network_tropical_same[tropical_false]) / np.sum(tropical_false))
    # Tropical false => Different label?
    print('Tropical false => Different label?')
    print(np.sum(network_tropical_different[tropical_false]) / np.sum(tropical_false))

    # network, Tropical false => Same label?
    print('network, Tropical false => Same label?')
    print(np.sum(network_tropical_same[network_tropical_false]) / np.sum(network_tropical_false))
    # network, Tropical false => Different label?
    print('network, Tropical false => Different label?')
    print(np.sum(network_tropical_different[network_tropical_false]) / np.sum(network_tropical_false))
    # network, Tropical Same label?
    print('network, Tropical Same label?')
    print(np.sum(network_labels == tropical_labels) / len(network_labels))
    # network, Tropical Different label?
    print('network, Tropical Same label?')
    print(np.sum(network_labels != tropical_labels) / len(network_labels))


def compute_averages():
    exp5_network_remaining_accuracy = np.zeros([6, 6])
    exp5_tropical_remaining_accuracy = np.zeros([6, 6])
    for i in range(5):
        save_dir = get_saving_directory(arg)
        exp5_network_remaining_accuracy += np.loadtxt(save_dir + "exp5_network_remaining_accuracy.csv",
                                                      delimiter=",")
        exp5_tropical_remaining_accuracy += np.loadtxt(save_dir + "exp5_tropical_remaining_accuracy.csv", delimiter=",")
    exp5_network_remaining_accuracy /= 5
    exp5_tropical_remaining_accuracy /= 5
    save_dir = get_saving_directory(arg)
    if not os.path.isdir(transformation_path):
        os.makedirs(transformation_path, exist_ok=True)
    np.savetxt(transformation_path + "exp5_network_remaining_accuracy.csv", exp5_network_remaining_accuracy,
               delimiter=",")
    np.savetxt(transformation_path + "exp5_tropical_remaining_accuracy.csv", exp5_tropical_remaining_accuracy,
               delimiter=",")


def extract_weight_matrices():
    def save_weight_matrix(layer, layer_type):
        if layer_type == 'conv2d':
            input_shape = layer.input_shape[1:]
            input_height, input_width, input_channels = input_shape
            output_shape = layer.output_shape[1:]
            output_height, output_width, output_channels = output_shape
            filter, bias = layer.get_weights()
            filter = filter  # .astype('float64') # (f_h, f_w, input_c, output_c)
            bias = bias  # .astype('float64')
            filter_height, filter_width = filter.shape[0:2]
            if filter_height == 1:  # 1x1 convolution
                filter = np.moveaxis(filter, source=[0, 1, 2, 3],
                                     destination=[1, 2, 3, 0])  # (output_c, f_h, f_w, input_c)
                row_length = input_width * input_channels
                filter = np.reshape(filter,
                                    newshape=[filter.shape[0], filter.shape[1], -1])  # (output_c, f_h, f_w * input_c)
                filter_flat = np.zeros(
                    [output_channels,
                     (filter_width * input_channels + (filter_height - 1) * row_length)])  # (output_c, input_c)
                for i in range(filter_height):
                    filter_flat[:, (i * row_length):(filter_width * input_channels + i * row_length)] = filter[:, i, :]
                filter_length = filter_flat.shape[1]
                A = np.zeros([output_height * output_width * output_channels,
                              1 + input_height * input_width * input_channels])
                for i in range(output_height):
                    for j in range(output_width):
                        A[((i * output_width + j) * output_channels):((i * output_width + j + 1) * output_channels),
                        1 + ((i * output_width + j) * input_channels):1 + (
                                (i * output_width + j) * input_channels + filter_length)] = filter_flat
                bias = np.tile(bias, output_height * output_width)
                A[:, 0] = bias
            elif filter_height == 3 and layer.strides[0] == 1:  # 3x3 convolution without stride
                padding_height, padding_width = filter_height - 2, filter_width - 2
                filter = np.moveaxis(filter, source=[0, 1, 2, 3],
                                     destination=[1, 2, 3, 0])  # (output_c, f_h, f_w, input_c)
                row_length = (input_width + 2 * padding_width) * input_channels
                filter = np.reshape(filter,
                                    newshape=[filter.shape[0], filter.shape[1], -1])  # (output_c, f_h, f_w * input_c)
                filter_flat = np.zeros(
                    [output_channels, (filter_width * input_channels + (filter_height - 1) * row_length)])
                for i in range(filter_height):
                    filter_flat[:, (i * row_length):(filter_width * input_channels + i * row_length)] = filter[:, i, :]
                filter_length = filter_flat.shape[1]

                A = np.zeros([output_height * output_width * output_channels,
                              (input_height + 2 * padding_height) * (
                                      input_width + 2 * padding_width) * input_channels + 1])
                for i in range(output_height):
                    for j in range(output_width):
                        A[((i * output_width + j) * output_channels):((i * output_width + j + 1) * output_channels),
                        1 + ((i * (input_width + 2) + j) * input_channels):1 + (
                                (i * (input_width + 2) + j) * input_channels + filter_length)] = filter_flat
                bias = np.tile(bias, output_height * output_width)
                A[:, 0] = bias
                indices = np.zeros([input_height + 2 * padding_height, input_width + 2 * padding_width, input_channels],
                                   dtype=bool)
                indices[1:-1, 1:-1, :] = True
                indices = np.reshape(indices, newshape=-1)
                indices = np.append(True, indices)
                A = A[:, indices]
            elif filter_height == 3 and layer.strides[0] == 2:  # 3x3 convolution with stride
                padding_height, padding_width = filter_height - 2, filter_width - 2
                filter = np.moveaxis(filter, source=[0, 1, 2, 3],
                                     destination=[1, 2, 3, 0])  # (output_c, f_h, f_w, input_c)
                row_length = (input_width + padding_width) * input_channels
                filter = np.reshape(filter,
                                    newshape=[filter.shape[0], filter.shape[1], -1])  # (output_c, f_h, f_w * input_c)
                filter_flat = np.zeros(
                    [output_channels, (filter_width * input_channels + (filter_height - 1) * row_length)])
                for i in range(filter_height):
                    filter_flat[:, (i * row_length):(filter_width * input_channels + i * row_length)] = filter[:, i, :]
                filter_length = filter_flat.shape[1]

                A = np.zeros([output_height * output_width * output_channels,
                              (input_height + padding_height) * (input_width + padding_width) * input_channels + 1])
                for i in range(output_height):
                    for j in range(output_width):
                        A[((i * output_width + j) * output_channels):((i * output_width + j + 1) * output_channels),
                        1 + ((2 * i * (input_width + padding_width) + 2 * j) * input_channels):1 + (
                                (2 * i * (
                                        input_width + padding_width) + 2 * j) * input_channels + filter_length)] = filter_flat
                bias = np.tile(bias, output_height * output_width)
                A[:, 0] = bias
                indices = np.zeros([input_height + 1, input_width + 1, input_channels], dtype=bool)
                indices[0:-1, 0:-1] = True
                indices = np.reshape(indices, newshape=-1)
                indices = np.append(True, indices)
                A = A[:, indices]
        elif layer_type == 'dense':
            W, bias = current_layer.get_weights()
            W = W.transpose()
            A = np.hstack([bias[:, np.newaxis], W])
        A = A.transpose()
        path = transformation_path + 'W/'
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        file_name = str(layer_idx) + '_' + network.layers[-(layer_idx + 2)].name + '.npy'
        np.save(path + file_name, A)

    for layer_idx in range(last_layer_index + 1):
        current_layer = network.layers[-(layer_idx + 2)]
        layer_type = current_layer.name.split('_')[0]
        if layer_type == 'conv2d' or layer_type == 'dense':
            save_weight_matrix(current_layer, layer_type)


def slide_and_evaluate_batch(pos_terms, data, v_shift, h_shift, filling='zeros'):
    bias = pos_terms[:, 0:1]
    main = pos_terms[:, 1:]
    main = main.reshape([-1, 32, 32, 3])
    main_shifted = shift_array(main, v_shift, h_shift, filling)
    shifted_terms = flatten_and_stack(main_shifted, bias)
    result = np.max(np.dot(shifted_terms, data), axis=0)
    return result


def slide_and_evaluate(pos_terms, data, v_shift, h_shift, filling='zeros'):
    result = [None] * no_labels
    for i in range(no_labels):
        result[i] = slide_and_evaluate_batch(pos_terms[i], data, v_shift, h_shift, filling=filling)
    return np.vstack(result)


def slide_extracted_function_over_image():
    logger = get_logger(arg)
    grouped_data, true_labels, network_labels = get_grouped_data(arg, network, 'test', get_labels=True)
    current_data = get_current_data(network, grouped_data, last_layer_index)
    pos_terms = load_tropical_function(arg, network)
    summing_result = evaluate_tropical_function(current_data, pos_terms)
    tropical_test_labels = np.argmax(summing_result, axis=0)
    maxing_result = np.copy(summing_result)
    tropical_accuracy = sum(tropical_test_labels == true_labels) / len(true_labels)
    shifts = [-1, 0, 1]
    for v_shift in shifts:
        for h_shift in shifts:
            if v_shift != 0 or h_shift != 0:
                for i in range(no_labels):
                    new_result = slide_and_evaluate(pos_terms, current_data, v_shift, h_shift)
                    summing_result += new_result
                    maxing_result = np.array([maxing_result, new_result])
                    maxing_result = np.max(maxing_result, axis=0)
    tropical_summing_test_labels = np.argmax(summing_result, axis=0)
    tropical_maxing_test_labels = np.argmax(maxing_result, axis=0)
    network_agreement_summing = sum(tropical_summing_test_labels == network_labels) / len(network_labels)
    logger.info(
        'Agreement of the tropical function and the neural network on the test set: ' + str(
            network_agreement_summing))
    test_accuracy_summing = sum(tropical_summing_test_labels == true_labels) / len(true_labels)
    logger.info(
        'Agreement of the tropical function and the true labels on the test set: ' + str(
            test_accuracy_summing))
    logger.info('Increase of test accuracy: ' + str(test_accuracy_summing / tropical_accuracy))
    network_agreement_maxing = sum(tropical_maxing_test_labels == network_labels) / len(network_labels)
    logger.info(
        'Agreement of the tropical function and the neural network on the test set: ' + str(
            network_agreement_maxing))
    test_accuracy_maxing = sum(tropical_maxing_test_labels == true_labels) / len(true_labels)
    logger.info(
        'Agreement of the tropical function and the true labels on the test set: ' + str(
            test_accuracy_maxing))
    logger.info('Increase of test accuracy: ' + str(test_accuracy_maxing / tropical_accuracy))


def slide_extracted_function_batchwise_over_image():
    logger = get_logger(arg)
    grouped_data, true_labels, network_labels = get_grouped_data(arg, network, 'test', get_labels=True)
    no_data_points = len(network_labels)
    current_data = get_current_data(network, grouped_data, last_layer_index)
    save_dir = get_tropical_function_directory(arg, network, last_layer_index, arg.data_type)
    shifts = [-2, -1, 0, 1, 2]
    no_shifts = len(shifts)
    result = np.zeros([no_labels, no_labels, no_shifts, no_shifts, no_data_points])
    no_batches = no_labels ** 2
    for batch_idx in range(no_batches):
        dim_number = batch_idx // no_labels
        data_group_number = batch_idx % no_labels
        pos_terms = load_tropical_function_batch(save_dir, batch_idx)
        result[dim_number, data_group_number, 0, 0] = evaluate_batch_of_tropical_function(current_data, pos_terms)
        for v_shift_counter in range(no_shifts):
            for h_shift_counter in range(no_shifts):
                if shifts[v_shift_counter] != 0 or shifts[h_shift_counter] != 0:
                    for i in range(no_labels):
                        result[dim_number, data_group_number, v_shift_counter, h_shift_counter] = \
                            slide_and_evaluate_batch(pos_terms, current_data, shifts[v_shift_counter],
                                                     shifts[v_shift_counter])
    result = np.max(result, axis=1)
    summing_result = np.sum(np.sum(result, axis=1), axis=1)
    maxing_result = np.max(np.max(result, axis=1), axis=1)

    tropical_test_labels = np.argmax(result[:, 0, 0], axis=0)
    tropical_accuracy = sum(tropical_test_labels == true_labels) / len(true_labels)
    tropical_summing_test_labels = np.argmax(summing_result, axis=0)
    tropical_maxing_test_labels = np.argmax(maxing_result, axis=0)

    network_agreement_summing = sum(tropical_summing_test_labels == network_labels) / len(network_labels)
    logger.info(
        'Agreement of the tropical function and the neural network on the test set: ' + str(
            network_agreement_summing))
    test_accuracy_summing = sum(tropical_summing_test_labels == true_labels) / len(true_labels)
    logger.info(
        'Agreement of the tropical function and the true labels on the test set: ' + str(
            test_accuracy_summing))
    logger.info('Increase of test accuracy: ' + str(test_accuracy_summing / tropical_accuracy))
    network_agreement_maxing = sum(tropical_maxing_test_labels == network_labels) / len(network_labels)
    logger.info(
        'Agreement of the tropical function and the neural network on the test set: ' + str(
            network_agreement_maxing))
    test_accuracy_maxing = sum(tropical_maxing_test_labels == true_labels) / len(true_labels)
    logger.info(
        'Agreement of the tropical function and the true labels on the test set: ' + str(
            test_accuracy_maxing))
    logger.info('Increase of test accuracy: ' + str(test_accuracy_maxing / tropical_accuracy))


def compute_coefficient_statistics(arg, fast):
    def calculate_statistics(value_dict, dict, i):
        for key in dict.keys():
            split_key = key.split('_')
            if split_key[0] == 'opt':
                dict[key]['mean'][i] = np.mean(value_dict[key][i])
                dict[key]['median'][i] = np.median(value_dict[key][i])
                dict[key]['std'][i] = np.std(value_dict[key][i])
                dict[key]['min'][i] = np.min(value_dict[key][i])
                dict[key]['max'][i] = np.max(value_dict[key][i])
            elif split_key[0] == 'all':
                for j in range(no_labels):
                    dict[key]['mean'][i][j] = np.mean(value_dict[key][i][j])
                    dict[key]['median'][i][j] = np.median(value_dict[key][i][j])
                    dict[key]['std'][i][j] = np.std(value_dict[key][i][j])
                    dict[key]['min'][i][j] = np.min(value_dict[key][i][j])
                    dict[key]['max'][i][j] = np.max(value_dict[key][i][j])
            elif split_key[0] == 'ranking':
                if len(split_key) > 2:
                    type = split_key[-2] + '_' + split_key[-1]
                else:
                    type = split_key[-1]
                sorted_values = np.sort(np.vstack(value_dict['all_' + type][i]), axis=0)
                indices = np.argmin(np.abs(sorted_values - value_dict['opt_' + type][i][np.newaxis, :]), axis=0)
                dict[key]['mean'][i] = np.mean(indices)
                dict[key]['median'][i] = np.median(indices)
                dict[key]['std'][i] = np.std(indices)
                dict[key]['min'][i] = np.min(indices)
                dict[key]['max'][i] = np.max(indices)
        return dict

    def create_dict(dim):
        if dim == 1:
            return {'mean': np.zeros(no_labels), 'median': np.zeros(no_labels), 'std': np.zeros(no_labels),
                    'min': np.zeros(no_labels), 'max': np.zeros(no_labels)}
        if dim == 2:
            return {'mean': np.zeros([no_labels, no_labels]), 'median': np.zeros([no_labels, no_labels]),
                    'std': np.zeros([no_labels, no_labels]), 'min': np.zeros([no_labels, no_labels]),
                    'max': np.zeros([no_labels, no_labels])}

    def get_current_indices(idx):
        if idx == 'all':
            current_data = stack_list_with_subgroups(grouped_test_data)
        else:
            current_data = grouped_test_data[idx]  # test data grouped according to labels predicted by the network
        current_data = np.reshape(current_data, newshape=[current_data.shape[0], -1])
        current_data = np.hstack([np.ones_like(current_data[:, 0:1]), current_data])
        current_data = current_data.transpose()
        return np.argmax(np.dot(pos_training_terms, current_data), axis=0)
        # for each test data point of index idx, returns the index of the training function on whose linear region
        # the data point lies

    def compute_distances(training_info, test_functions, type):
        if type == '1-1':
            training_functions = training_terms[training_info]
            distances = compute_1_1_euclidean_distances(training_functions, test_functions)
            angles = compute_1_1_angles(training_functions, test_functions)
        elif type == 'pairwise':
            training_functions = training_info
            distances = cdist(training_functions, test_functions)
            angles = np.arccos(1 - cdist(training_functions, test_functions, 'cosine'))
        return distances, angles

    pos_training_terms, neg_training_terms = load_tropical_function(arg, network, 'training', epoch_number,
                                                                    load_negative=True, stacked=True)
    pos_test_terms, neg_test_terms = load_tropical_function(arg, network, 'test', epoch_number, load_negative=True,
                                                            stacked=True)
    training_terms = pos_training_terms - neg_training_terms
    test_terms = pos_test_terms - neg_test_terms

    grouped_test_data, true_labels, network_labels = get_grouped_data(arg, network, data_type='test', get_labels=True)
    max_train_func_idxs = get_current_indices('all')
    tropical_test_labels = get_tropical_test_labels(arg, network, grouped_test_data, last_layer_index, epoch_number)
    tropical_true_indices = (tropical_test_labels == network_labels)
    tropical_false_indices = np.logical_not(tropical_true_indices)
    true_test_functions = test_terms[tropical_true_indices]
    true_max_train_func_idxs = max_train_func_idxs[tropical_true_indices]
    false_test_functions = test_terms[tropical_false_indices]
    false_max_train_func_idxs = max_train_func_idxs[tropical_false_indices]
    true_test_labels = network_labels[tropical_true_indices]
    false_test_labels = network_labels[tropical_false_indices]
    true_test_functions = group_points(true_test_functions, true_test_labels, no_labels)
    false_test_functions = group_points(false_test_functions, false_test_labels, no_labels)
    true_max_train_func_idxs = group_points(true_max_train_func_idxs, true_test_labels, no_labels)
    false_max_train_func_idxs = group_points(false_max_train_func_idxs, false_test_labels, no_labels)
    no_true_points = np.zeros([no_labels])
    no_false_points = np.zeros([no_labels])
    for i in range(no_labels):
        no_true_points[i] = true_max_train_func_idxs[i].shape[0]
        no_false_points[i] = false_max_train_func_idxs[i].shape[0]

    if fast:
        stats_dict = {'opt_angles': create_dict(1), 'opt_distances': create_dict(1),
                      'opt_true_angles': create_dict(1), 'opt_true_distances': create_dict(1),
                      'opt_false_angles': create_dict(1), 'opt_false_distances': create_dict(1),
                      'no_true_points': no_true_points, 'no_false_points': no_false_points}
    else:
        stats_dict = {'opt_angles': create_dict(1), 'opt_distances': create_dict(1),
                      'opt_true_angles': create_dict(1), 'opt_true_distances': create_dict(1),
                      'opt_false_angles': create_dict(1), 'opt_false_distances': create_dict(1),
                      'all_angles': create_dict(2), 'all_distances': create_dict(2),
                      'ranking_angles': create_dict(1), 'ranking_distances': create_dict(1),
                      'ranking_true_angles': create_dict(1), 'ranking_true_distances': create_dict(1),
                      'ranking_false_angles': create_dict(1), 'ranking_false_distances': create_dict(1),
                      'no_true_points': no_true_points, 'no_false_points': no_false_points}

    all_angles = [[None] * 10 for i in range(10)]
    all_true_angles = [[None] * 10 for i in range(10)]
    all_false_angles = [[None] * 10 for i in range(10)]
    all_distances = [[None] * 10 for i in range(10)]
    all_true_distances = [[None] * 10 for i in range(10)]
    all_false_distances = [[None] * 10 for i in range(10)]
    opt_true_distances = [None] * 10
    opt_true_angles = [None] * 10
    opt_false_distances = [None] * 10
    opt_false_angles = [None] * 10
    opt_distances = [None] * 10
    opt_angles = [None] * 10
    for i in range(no_labels):
        opt_true_distances[i], opt_true_angles[i] = compute_distances(true_max_train_func_idxs[i],
                                                                      true_test_functions[i], '1-1')
        opt_false_distances[i], opt_false_angles[i] = compute_distances(false_max_train_func_idxs[i],
                                                                        false_test_functions[i], '1-1')
        opt_distances[i] = np.hstack([opt_true_distances[i], opt_false_distances[i]])
        opt_angles[i] = np.hstack([opt_true_angles[i], opt_false_angles[i]])
        if not fast:
            for j in range(no_labels):
                training_functions_j = load_functions('training', j)
                all_true_distances[i][j], all_true_angles[i][j] = compute_distances(training_functions_j,
                                                                                    true_test_functions[i], 'pairwise')
                all_false_distances[i][j], all_false_angles[i][j] = compute_distances(training_functions_j,
                                                                                      false_test_functions[i],
                                                                                      'pairwise')
                all_distances[i][j] = np.hstack([all_true_distances[i][j], all_false_distances[i][j]])
                all_angles[i][j] = np.hstack([all_true_angles[i][j], all_false_angles[i][j]])
        value_dict = {'opt_angles': opt_angles, 'opt_distances': opt_distances,
                      'opt_true_angles': opt_true_angles, 'opt_true_distances': opt_true_distances,
                      'opt_false_angles': opt_false_angles, 'opt_false_distances': opt_false_distances,
                      'all_angles': all_angles, 'all_distances': all_distances,
                      'all_true_angles': all_true_angles, 'all_true_distances': all_true_distances,
                      'all_false_angles': all_false_angles, 'all_false_distances': all_false_distances}
        stats_dict = calculate_statistics(value_dict, stats_dict, i)

    save_dir = get_saving_directory(arg)
    saving_path = os.path.join(save_dir, 'Coefficient_Analysis')
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    file_name = 'coefficient_statistics.pkl'
    with open(os.path.join(saving_path, file_name), 'wb') as f:
        pickle.dump(stats_dict, f)

    for key in value_dict.keys():
        split_key = key.split('_')
        current_saving_path = os.path.join(saving_path, split_key[0])
        if not os.path.isdir(current_saving_path):
            os.makedirs(current_saving_path, exist_ok=True)
        if split_key[0] == 'opt':
            for i in range(no_labels):
                file_name = key + '_' + str(i)
                np.save(os.path.join(current_saving_path, file_name), value_dict[key][i])
        elif split_key[0] == 'all' and not fast:
            for i in range(no_labels):
                for j in range(no_labels):
                    file_name = key + '_' + str(i) + '_' + str(j)
                    np.save(os.path.join(current_saving_path, file_name), value_dict[key][i][j])

    loading_path = os.path.join(saving_path, 'opt')
    true_angles = [None] * no_labels
    false_angles = [None] * no_labels
    true_distances = [None] * no_labels
    false_distances = [None] * no_labels
    for i in range(no_labels):
        true_angles[i] = np.load(os.path.join(loading_path, 'opt_true_angles_' + str(i) + '.npy'))
        false_angles[i] = np.load(os.path.join(loading_path, 'opt_false_angles_' + str(i) + '.npy'))
        true_distances[i] = np.load(os.path.join(loading_path, 'opt_true_distances_' + str(i) + '.npy'))
        false_distances[i] = np.load(os.path.join(loading_path, 'opt_false_distances_' + str(i) + '.npy'))

    true_angles = np.hstack(true_angles)
    false_angles = np.hstack(false_angles)
    true_distances = np.hstack(true_distances)
    false_distances = np.hstack(false_distances)

    savemat(loading_path + arg.network_type_fine + '_' + arg.network_number + '_angles_distances.mat',
            dict(true_angles=true_angles, false_angles=false_angles,
                 true_distances=true_distances, false_distances=false_distances))


def compare_linear_functions():
    def correlation(A, B):
        def subtract_mean(A):
            return A - np.mean(A, axis=0)[np.newaxis, :]

        A = subtract_mean(A)
        B = subtract_mean(B)
        A_B = np.sum(A * B, axis=0)
        A_A = np.sum(A * A, axis=0)
        B_B = np.sum(B * B, axis=0)
        divisor = np.sqrt(A_A) * np.sqrt(B_B)
        correlation = np.divide(A_B, divisor, out=np.ones_like(A_B), where=(divisor != 0))
        if epoch_number == '00':
            correlation[0] = 0
        return correlation

    arg.network_number = network_number
    pos_terms_0, neg_terms_0 = load_tropical_function(arg, network, last_layer_index, no_labels, arg.data_type,
                                                      epoch_number, load_negative=True)
    terms_0 = np.vstack(pos_terms_0) - np.vstack(neg_terms_0)
    arg.network_number = network_number_2
    pos_terms_1, neg_terms_1 = load_tropical_function(arg, network, last_layer_index, no_labels, arg.data_type,
                                                      epoch_number, load_negative=True)
    terms_1 = np.vstack(pos_terms_1) - np.vstack(neg_terms_1)
    # terms_0 = terms_0[0:no_data_points]
    # terms_1 = terms_1[0:no_data_points]
    return correlation(terms_0, terms_1)


def compare_activation_patterns():
    def compute_activation_pattern_agreement(x_test, compute_activation_patterns_size=True):
        x_train_associated = [None] * no_labels
        for i in range(no_labels):
            x_train_associated[i] = get_associated_training_points(arg, network, x_test[i], x_train)
        test_activation_patterns = get_activation_patterns(arg, network, max_data_group_size, grouped_data=x_test)
        associated_training_activation_patterns = \
            get_activation_patterns(arg, network, max_data_group_size, grouped_data=x_train_associated)
        no_relevant_layers = len(test_activation_patterns[0][0])
        activation_patterns_agreement = [None] * no_relevant_layers
        for k in range(no_relevant_layers):
            activation_patterns_agreement[k] = []
            for i in range(no_labels):
                no_subgroups = len(test_activation_patterns[i])
                for j in range(no_subgroups):
                    indices = (test_activation_patterns[i][j][k] == associated_training_activation_patterns[i][j][k])
                    agreement = np.squeeze(np.apply_over_axes(np.sum, indices, range(1, indices.ndim)))
                    activation_patterns_agreement[k].append(agreement)
            activation_patterns_agreement[k] = np.hstack(activation_patterns_agreement[k])
        agreements = {}
        agreements['max'] = np.array(list(map(lambda x: np.max(x), activation_patterns_agreement)))
        agreements['mean'] = np.array(list(map(lambda x: np.mean(x), activation_patterns_agreement)))
        agreements['min'] = np.array(list(map(lambda x: np.min(x), activation_patterns_agreement)))
        if compute_activation_patterns_size:
            activation_patterns_size = np.zeros(no_relevant_layers)
            for k in range(no_relevant_layers):
                current_layer_shape = test_activation_patterns[0][0][k].shape[1:]
                activation_patterns_size[k] = reduce(lambda x, y: x * y, current_layer_shape)
            return agreements, activation_patterns_size
        return agreements

    def compute_percentage(agreements, activation_patterns_size):
        percentages = {}
        for key in agreements.keys():
            percentages[key] = np.flip(agreements[key] / activation_patterns_size)
        return percentages

    def print_dict(Dict):
        def print_array(array):
            for element in array:
                print(round(element, 2), end=' & ')
            print()

        for key in Dict.keys():
            print_array(Dict[key])

    max_data_group_size = get_max_data_group_size(arg)
    x_train = get_grouped_data(arg, network, max_data_group_size, data_type='training')
    x_train = stack_list_with_subgroups(x_train)
    x_test, true_labels, network_labels = get_grouped_data(arg, network, max_data_group_size=max_data_group_size,
                                                           data_type='test', get_labels=True)
    tropical_test_labels = get_tropical_test_labels(arg, network, x_test, last_layer_index, epoch_number)
    indices = (tropical_test_labels == network_labels)
    grouped_indices = group_points(indices, network_labels, no_labels, max_data_group_size)
    x_test_right, x_test_wrong = partition_according_to_correct_indices(x_test, grouped_indices)
    agreements, activation_patterns_size = compute_activation_pattern_agreement(x_test)
    percentages = compute_percentage(agreements, activation_patterns_size)
    print_dict(percentages)
    agreements = compute_activation_pattern_agreement(x_test_right, compute_activation_patterns_size=False)
    percentages = compute_percentage(agreements, activation_patterns_size)
    print_dict(percentages)
    agreements = compute_activation_pattern_agreement(x_test_wrong, compute_activation_patterns_size=False)
    percentages = compute_percentage(agreements, activation_patterns_size)
    print_dict(percentages)


def save_to_mat():
    save_dir = get_saving_directory(arg)
    transformation_path = os.path.join(save_dir, 'Coefficient_Analysis', 'opt')

    no_labels = get_no_labels(network)
    true_angles = [None] * no_labels
    false_angles = [None] * no_labels
    true_distances = [None] * no_labels
    false_distances = [None] * no_labels
    for i in range(no_labels):
        true_angles[i] = np.load(os.path.join(transformation_path, 'opt_true_angles_' + str(i) + '.npy'))
        false_angles[i] = np.load(os.path.join(transformation_path, 'opt_false_angles_' + str(i) + '.npy'))
        true_distances[i] = np.load(os.path.join(transformation_path, 'opt_true_distances_' + str(i) + '.npy'))
        false_distances[i] = np.load(os.path.join(transformation_path, 'opt_false_distances_' + str(i) + '.npy'))

    true_angles = np.hstack(true_angles)
    false_angles = np.hstack(false_angles)
    true_distances = np.hstack(true_distances)
    false_distances = np.hstack(false_distances)

    savemat(
        os.path.join(transformation_path, arg.network_type_fine + '_' + arg.network_number + '_angles_distances.mat'),
        dict(true_angles=true_angles, false_angles=false_angles,
             true_distances=true_distances, false_distances=false_distances))


def append_coefficient_statistics(arg, epoch_number):
    pos_terms, neg_terms = load_tropical_function(arg, network, arg.data_type, epoch_number, load_negative=True)
    current_selected_terms = []
    max_no_selected_terms_per_label = 100
    for i in range(len(pos_terms)):
        no_selected_terms = np.min([pos_terms[i].shape[0], max_no_selected_terms_per_label])
        preliminary_terms = pos_terms[i][0:no_selected_terms] - neg_terms[i][0:no_selected_terms]
        preliminary_terms = preliminary_terms[:, 1:].reshape([-1, 32, 32, 3])
        current_selected_terms.append(preliminary_terms)
    selected_terms.append(current_selected_terms)
    terms = np.vstack(pos_terms) - np.vstack(neg_terms)
    means.append(np.mean(terms, axis=0))
    medians.append(np.median(terms, axis=0))
    stds.append(np.std(terms, axis=0))
    mins.append(np.min(terms, axis=0))
    maxs.append(np.max(terms, axis=0))


def interpolation(arg, activation_pattern_changes=True, label_changes=False):
    def compute_label_changes():
        predicted_capped_labels = np.argmax(network.predict(modified_capped_image), axis=1)
        original_label = predicted_capped_labels[0]
        first_label_change = np.argmin(predicted_capped_labels == original_label)
        first_label_changes.append(first_label_change)
        distances_at_first_label_changes(steps[0, first_label_change])

    def compute_AP_changes():
        def reshape_activation_patterns(activation_patterns):
            reshaped_activation_patterns = []
            for activation_pattern in activation_patterns:
                reshaped_activation_patterns.append(
                    np.reshape(activation_pattern, [different_im_per_batch, no_steps, -1]))
            return reshaped_activation_patterns

        activation_patterns = get_activation_patterns(arg, network, data_group=modified_capped_image)
        activation_patterns = reshape_activation_patterns(activation_patterns)
        for i in range(different_im_per_batch):
            first_AP_change = no_steps
            for activation_pattern in activation_patterns:
                unique_activation_pattern, indices = np.unique(activation_pattern[i][0:first_AP_change],
                                                               return_index=True, axis=0)
                first_AP_change = np.partition(np.append(indices, first_AP_change), 1)[1]
            first_AP_changes.append(first_AP_change)
            distances_at_first_AP_changes.append(steps[0, min(first_AP_change, no_steps - 1)])

    save_dir = get_saving_directory(arg)
    data_types = ['random_1000', 'random_10000']  # ['training', 'test', 'random', 'random_100']
    statistics = np.zeros([11 * len(data_types)])
    for j in range(len(data_types)):
        arg.data_type = data_types[j]
        images, labels = load_data(arg, arg.data_type)
        if label_changes:
            first_label_changes, distances_at_first_label_changes = [], []
            steps = 10 ** np.linspace(-7, 1.5, 100) - 10 ** (-7)
        else:
            steps = 10 ** np.linspace(-5, 4, 100) - 10 ** (-5)
        if activation_pattern_changes:
            first_AP_changes, distances_at_first_AP_changes = [], []
        steps = steps[np.newaxis, :]
        no_steps = steps.size
        no_images = 500
        batch_size = 5000
        total_no_images = no_images * no_steps
        different_im_per_batch = int(batch_size / no_steps)
        no_iterations = int(np.ceil(total_no_images / batch_size))
        for i in range(no_iterations):  # x_train.shape[0]
            image = images[i * different_im_per_batch:(i + 1) * different_im_per_batch].reshape(
                [different_im_per_batch, -1])[:, np.newaxis]
            d = np.random.rand(different_im_per_batch, 3072) * 2 - 1
            d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]
            d = d[:, :, np.newaxis]
            d_steps = np.dot(d, steps)
            d_steps = np.moveaxis(d_steps, [0, 1, 2], [0, 2, 1])
            modified_image = image + d_steps
            modified_image = np.reshape(modified_image, [-1, 32, 32, 3])
            modified_capped_image = np.clip(modified_image, 0, 1)
            if label_changes:
                compute_label_changes()
            if activation_pattern_changes:
                compute_AP_changes()
        Dict = {}
        if label_changes:
            Dict['first_label_changes'] = np.array(first_label_changes)
            Dict['distances_at_first_label_changes'] = np.array(distances_at_first_label_changes)
            Dict['no_no_label_changes'] = np.sum(first_label_changes == 0)
            Dict['distances_at_first_label_changes'] = Dict['distances_at_first_label_changes'][
                first_label_changes != 0]
            Dict['first_label_changes'] = Dict['first_label_changes'][first_label_changes != 0]
        if activation_pattern_changes:
            Dict['first_AP_changes'] = np.array(first_AP_changes)
            Dict['distances_at_first_AP_changes'] = np.array(distances_at_first_AP_changes)
            statistics[0 + 11 * j] = np.min(Dict['first_AP_changes'])
            statistics[1 + 11 * j] = np.mean(Dict['first_AP_changes'])
            statistics[2 + 11 * j] = np.std(Dict['first_AP_changes'])
            statistics[3 + 11 * j] = np.max(Dict['first_AP_changes'])
            statistics[4 + 11 * j] = np.min(Dict['distances_at_first_AP_changes'])
            statistics[5 + 11 * j] = np.mean(Dict['distances_at_first_AP_changes'])
            statistics[6 + 11 * j] = np.exp(np.mean(np.log(Dict['distances_at_first_AP_changes'])))
            statistics[7 + 11 * j] = np.exp(np.std(np.log(Dict['distances_at_first_AP_changes'])))
            statistics[8 + 11 * j] = np.std(Dict['distances_at_first_AP_changes'])
            statistics[9 + 11 * j] = np.max(Dict['distances_at_first_AP_changes'])
            accuracy = np.sum(np.argmax(network.predict(images), axis=1) == labels) / labels.size
            statistics[10 + 11 * j] = accuracy

        file_name = arg.data_type + '_interpolation.pkl'
        path = os.path.join(save_dir, file_name)
        with open(path, 'wb') as f:
            pickle.dump(Dict, f)
    np.savetxt(os.path.join(save_dir, 'interpolation_statistics.csv'), statistics, delimiter=",", fmt='%s')


def compute_network_accuracies(network, x_train, y_train, x_test, y_test, outputs):
    def compute_accuracy(network, data, true_labels):
        network_labels = np.argmax(network.predict(data), axis=1)
        accuracy = np.sum(network_labels == true_labels) / true_labels.size

    outputs['training'] = compute_accuracy(network, x_train, y_train)
    outputs['test'] = compute_accuracy(network, x_test, y_test)


logger = get_logger(arg)
epoch_numbers = get_epoch_numbers(arg)
no_data_points = 10
epoch_numbers = ['00', '01', '02']

if arg.mode == 'save_linear_coefficients_to_mat':
    means = []
    medians = []
    stds = []
    mins = []
    maxs = []
    selected_terms = []

if arg.mode == 'exp11_compare_linear_functions':
    angles = []
    correlations = []
    distances = []
    network_number = arg.network_number
    network_number_2 = arg.network_number_2
    max_data_group_size = get_max_data_group_size(arg)
    network = load_network(arg, '00')
    last_layer_index = get_last_layer_index(network)
    no_labels = get_no_labels(network)

if arg.mode == 'compute_network_accuracies':
    training_accuracies = []
    test_accuracies = []
    x_train, y_train, x_test, y_test = load_data(arg)

for epoch_number in epoch_numbers:
    print('Epoch number: ' + str(epoch_number))
    if not arg.mode == 'exp11_compare_linear_functions':
        network = load_network(arg, epoch_number)
        last_layer_index = get_last_layer_index(network)
        no_labels = get_no_labels(network)
    if arg.mode == 'exp1_count':
        no_linear_regions = count_no_linear_regions()
        print(no_linear_regions)
    elif arg.mode == 'exp2_val':
        val_accuracy_and_false_classification()
    elif arg.mode == 'exp3_compare':
        compare_test_and_training_images(arg, lower_true_label=arg.lower_label, idx=arg.idx)
    elif arg.mode == 'exp4_variation':
        random_training_sampling()
    elif arg.mode == 'exp5_compute_translation_invariance':
        compute_translation_invariance(shift_type='nearest')
        compute_translation_invariance(shift_type='zeros')
    elif arg.mode == 'exp6_add_shifted_functions':
        add_translation_invariance(shift_type='nearest')
        add_translation_invariance(shift_type='zeros')
    elif arg.mode == 'exp7_implications':
        check_implications()
    elif arg.mode == 'exp8_extract_weight_matrices':
        extract_weight_matrices()
    elif arg.mode == 'exp9_compute_coefficient_statistics':
        compute_coefficient_statistics(arg, arg.fast)
    elif arg.mode == 'exp10_slide_extracted_function_over_image':
        if arg.extract_all_dimensions:
            slide_extracted_function_batchwise_over_image()
        else:
            slide_extracted_function_over_image()
    elif arg.mode == 'exp11_compare_linear_functions':
        correlations.append(compare_linear_functions())
    elif arg.mode == 'compute_network_accuracies':
        with Manager() as manager:
            outputs = manager.dict()
            compute_network_accuracies(network, x_train, y_train, x_test, y_test, outputs)
            p = Process(target=compute_network_accuracies, args=(network, x_train, y_train, x_test, y_test, outputs))
            p.start()
            p.join()
            training_accuracies.append(outputs['training'])
            test_accuracies.append(outputs['test'])
    elif arg.mode == 'exp12_compare_activation_patterns':
        compare_activation_patterns()
    elif arg.mode == 'exp14_interpolation':
        interpolation(arg)
    elif arg.mode == 'compute_averages':
        compute_averages()
    elif arg.mode == 'save_to_mat':
        save_to_mat()
    elif arg.mode == 'save_linear_coefficients_to_mat':
        append_coefficient_statistics(arg, epoch_number)

if arg.mode == 'compute_network_accuracies':
    training_accuracies = np.vstack(angles)
    test_accuracies = np.vstack(correlations)
    network_accuracies = np.hstack([training_accuracies, test_accuracies])
    saving_directory = get_saving_directory(arg)
    path = os.path.join(saving_directory_without_network_number, 'network_accuracies.csv')
    np.savetxt(path, network_accuracies, delimiter=",")

if arg.mode == 'exp11_compare_linear_functions':
    correlations = np.vstack(correlations)
    directory = create_directory('/home/martin/Documents/MATLAB/tropex/Data/Exp11', 'LATEST_VERSION')
    if arg.data_set == 'MNIST':
        architecture = arg.network_type_fine
    else:
        architecture = arg.network_type_coarse
    if arg.network_number_2 == '1':
        ending = '1.mat'
    elif arg.network_number_2 == '3':
        ending = '2.mat'
    file_name = '_'.join([arg.data_set, architecture, arg.data_type, ending])
    path = os.path.join(directory, file_name)
    savemat(path, {'correlations': correlations})

if arg.mode == 'save_linear_coefficients_to_mat':
    def reshape_to_image(array):
        array = np.array(array)
        # bias = array[:, 0]
        coefficients = array[:, 1:]
        coefficients = np.reshape(coefficients, [-1, 32, 32, 3])
        return coefficients


    matlab_directory = '/home/martin/Documents/MATLAB/Tropex/Data/Exp13'
    means = reshape_to_image(means)
    medians = reshape_to_image(medians)
    stds = reshape_to_image(stds)
    mins = reshape_to_image(mins)
    maxs = reshape_to_image(maxs)
    Dict = {'means': means, 'medians': medians, 'stds': stds, 'mins': mins, 'maxs': maxs,
            'selected_terms': selected_terms}
    savemat(os.path.join(matlab_directory, 'coefficient_statistics.mat'), Dict)

print_end(start_time)
