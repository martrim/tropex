import nvidia_smi  # can be installed via 'pip install nvidia-ml-py3'
import numpy as np
import os
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from Utilities.Custom_Settings import configure_gpu
from Utilities.Data_Loader import load_data
from Utilities.Saver import create_directory, get_saving_directory

# To raise an exception on runtime warnings, used for debugging.
np.seterr(all='raise')


def compute_maximal_difference(array1, array2):
    return np.max(np.abs(array1 - array2))


def evaluate_batch_of_tropical_function(data, pos_terms, neg_terms=None):
    pos_result = np.max(np.dot(pos_terms, data), axis=0)
    if neg_terms is not None:
        neg_result = np.max(np.dot(neg_terms, data), axis=0)
        return pos_result, neg_result
    return pos_result


def evaluate_tropical_function(data, labels, pos_terms, neg_terms=None):
    # Input:
    # (data_size, no_data_points)-sized array data
    # (no_labels)-sized lists pos_terms, neg_terms
    # Output:
    # (no_labels, no_data_points)-sized array

    def compute_result(terms):
        result = [-np.ones(no_data_points)] * no_labels
        for label in range(no_labels):
            current_terms = terms[labels == label]
            if len(current_terms) > 0:
                result[label] = np.max(np.dot(current_terms[:, 1:], data) + current_terms[:, 0:1], axis=0)
        return result

    no_labels = int(np.max(labels) + 1)
    no_data_points = data.shape[1]
    pos_result = compute_result(pos_terms)
    if neg_terms is not None:
        neg_result = compute_result(neg_terms)
        return np.vstack(pos_result), np.vstack(neg_result)
    return np.vstack(pos_result)


def evaluate_network_on_subgrouped_data(network, data_batches):
    output_predictor = K.function([network.layers[0].input],
                                  [network.layers[-2].output])
    x_train_predicted = []
    for data_batch in data_batches:
        x_train_predicted.append(np.max(output_predictor([data_batch])[0], axis=1))
    x_train_predicted = np.hstack(x_train_predicted)
    return x_train_predicted


def get_current_data(grouped_data, layer_idx, network=None, no_data_groups=None):
    if layer_idx == 0:
        current_data = np.vstack(grouped_data)
    else:
        if no_data_groups is None:
            no_data_groups = len(grouped_data)
        current_data = [None] * no_data_groups
        for batch_idx in range(no_data_groups):
            current_data[batch_idx] = predict_data_batchwise(network, grouped_data[batch_idx], 'no_split',
                                                             layer_idx + 3)
        current_data = np.vstack(current_data)
    return prepare_data_for_tropical_function(current_data)


def get_max_data_group_size():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    total_memory = info.total
    if total_memory >= 12 * (10 ** 9):
        return 2 ** 12
    elif total_memory >= 6 * (10 ** 9):
        return 2 ** 11
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return 2 ** 12


def get_last_layer_index(network):
    return len(network.layers) - 2


def get_no_labels(network):
    return network.layers[-1].output_shape[1]


def partition_according_to_correct_indices(x_test, indices):
    no_groups = len(x_test)
    x_test_right = [None] * no_groups
    x_test_wrong = [None] * no_groups
    for i in range(no_groups):
        no_subgroups = len(x_test[i])
        x_test_right[i] = [None] * no_subgroups
        x_test_wrong[i] = [None] * no_subgroups
        for j in range(no_subgroups):
            x_test_right[i][j] = x_test[i][j][indices[i][j]]
            x_test_wrong[i][j] = x_test[i][j][np.logical_not(indices[i][j])]
    return x_test_right, x_test_wrong


def reorder_terms(terms, old_labels, new_labels, no_labels, max_data_group_size):
    old_labels = np.squeeze(np.vstack(old_labels))
    new_labels = np.squeeze(np.vstack(new_labels))
    terms = np.vstack(terms)
    new_labels = group_points(new_labels, old_labels, no_labels, max_data_group_size)
    new_labels = stack_list_with_subgroups(new_labels)
    terms = group_points(terms, new_labels, no_labels, max_data_group_size)
    return terms


def group_points(points, labels, no_labels, max_data_group_size=None):
    def ceiling_division(a, b):
        return -(-a // b)

    print('GROUPING POINTS!')
    grouped_points = [None] * no_labels
    for i in range(no_labels):
        grouped_points[i] = points[labels == i]
        if max_data_group_size is not None:
            group_size = grouped_points[i].shape[0]
            no_subgroups = max(ceiling_division(group_size, max_data_group_size), 1)
            grouped_points[i] = np.array_split(grouped_points[i], no_subgroups)
    return grouped_points


def make_batches(points, max_data_group_size):
    def ceiling_division(a, b):
        return -(-a // b)

    no_batches = max(ceiling_division(points.shape[0], max_data_group_size), 1)
    batches = np.array_split(points, no_batches)
    return batches


def get_batch_data(arg, network, data_type=None):
    if data_type is None:
        data_type = arg.data_type
    data, true_labels = load_data(arg, data_type)
    if arg.data_type == 'training' and arg.data_set == 'CIFAR10':
        data = data[arg.data_points_lower:arg.data_points_upper]
        true_labels = true_labels[arg.data_points_lower:arg.data_points_upper]

    network_labels = np.argmax(network.predict(data), axis=1)
    max_data_group_size = get_max_data_group_size()
    data_batches = make_batches(data, max_data_group_size)
    true_labels = make_batches(true_labels, max_data_group_size)
    network_labels = make_batches(network_labels, max_data_group_size)
    return data_batches, true_labels, network_labels


def stack_list_with_subgroups(terms):
    stacked_list = []
    for term in terms:
        for subterm in term:
            stacked_list.append(subterm)
    return np.concatenate(stacked_list)


def get_max_indices(terms, test_data):
    no_batches = len(test_data)
    indices = [None] * no_batches
    for batch_idx in range(no_batches):
        test_data_batch = prepare_data_for_tropical_function(test_data[batch_idx])
        indices[batch_idx] = np.argmax(np.dot(terms[:, 1:], test_data_batch) + terms[:, 0:1], axis=0)
    return np.concatenate(indices)


def get_tropical_test_labels(terms, labels, test_data):
    return labels[get_max_indices(terms, test_data)]


def load_tropical_function_batch(save_dir, batch_idx, load_negative=False):
    pos_terms = np.load(os.path.join(save_dir, 'pos_label_' + str(batch_idx) + '.npy'))
    if load_negative:
        neg_terms = np.load(os.path.join(save_dir, 'neg_label_' + str(batch_idx) + '.npy'))
        return pos_terms, neg_terms
    return pos_terms


def get_tropical_filename_ending(batch_idx):
    return '_batch_{}.npy'.format(batch_idx)


def get_no_subgroups(list_of_file_names, data_group_number):
    return len(list(filter(lambda file_name: 'pos_label_' + str(data_group_number) in file_name, list_of_file_names)))


def get_folder_name(network, index):
    return '_'.join([str(index), network.layers[index].name])


def load_tropical_function(arg, folder_name, data_type, epoch_number, sign='pos'):
    save_dir = get_tropical_function_directory(arg, folder_name, data_type, epoch_number)
    no_batches = max(map(lambda x: int(''.join(filter(str.isdigit, x))), os.listdir(save_dir))) + 1
    terms = [None] * no_batches
    for batch_idx in range(no_batches):
        file_name_ending = get_tropical_filename_ending(batch_idx)
        path = os.path.join(save_dir, sign + file_name_ending)
        if not os.path.isfile(path):
            continue
        else:
            terms[batch_idx] = np.load(path)
    terms = np.vstack(list(filter(None.__ne__, terms)))
    true_labels = terms[:, 0]
    network_labels = terms[:, 1]
    return terms[:, 2:], true_labels, network_labels
    

def predict_data(network, data_batch, flag, layer_idx):
    # predictor = K.function([network.layers[0].input],
    #                        [network.layers[-index].output])
    # predictor([data])[0]
    predictor = Model(inputs=network.layers[0].input, outputs=network.layers[-layer_idx].output)
    if flag == 'no_split':
        return predictor.predict(data_batch)
    elif flag == 0:
        return predictor.predict(data_batch[0])
    else:
        return predictor.predict(data_batch[1])


def predict_data_batchwise(network, data_batch, flag, layer_idcs):
    predictor = K.function([network.input], [network.layers[-layer_idcs].output])
    if flag == 'no_split':
        return predictor([data_batch])[0]
    elif flag == 0:
        return predictor([data_batch[0]])[0]
    else:
        return predictor([data_batch[1]])[0]


def predict_activation_patterns_batchwise(network, data_batch, flag, layer_idcs):
    network_output = predict_data_batchwise(network, data_batch, flag, layer_idcs)
    activation_patterns = list(map(lambda x: x > 0, network_output))
    return activation_patterns


def get_layer_type(layer):
    return layer.name.split('_')[0]


def get_activation_patterns(arg, network, batch=None):
    # activation_patterns = no_data_groups x no_data_subgroups x no_relevant_layers
    def turn_data_into_activation_patterns(data):
        current_activation_patterns = []
        for layer in reversed(layers_without_softmax):
            layer_type = get_layer_type(layer)
            if layer_type in ['leaky', 're', 'activation']:
                data_after_layer = data.pop()
                current_activation_patterns.append(data_after_layer <= 0)
            elif layer_type == 'max':
                data_after_layer = data.pop()
                data_before_layer = data.pop()
                repeated_data = np.repeat(np.repeat(data_after_layer, repeats=2, axis=1), repeats=2, axis=2)

                def pad_data_before_layer(data_before_layer, repeated_data, axis):
                    shape = data_before_layer.shape
                    if repeated_data.shape[axis] == shape[axis] + 1:
                        new_array = np.full(shape[0:axis] + (1,) + shape[axis+1:], np.NINF)
                        data_before_layer = np.concatenate([data_before_layer, new_array], axis=axis)
                        padding = True
                    else:
                        padding = False
                    return data_before_layer, padding

                data_before_layer, vertical_padding = pad_data_before_layer(data_before_layer, repeated_data, 1)
                data_before_layer, horizontal_padding = pad_data_before_layer(data_before_layer, repeated_data, 2)

                aps = (repeated_data != data_before_layer)
                # The following guarantees that in each maxpooling square, at least 3 out of 4 elements of B are set to zero.
                if vertical_padding:
                    aps[:, -1, :, :] = True
                if horizontal_padding:
                    aps[:, :, -1, :] = True
                aps[:, ::2, 1::2, :] = np.logical_or(aps[:, ::2, 1::2, :], np.logical_not(aps[:, ::2, ::2]))
                aps[:, 1::2, ::2, :] = np.logical_or(aps[:, 1::2, ::2, :], np.logical_not(aps[:, ::2, ::2]))
                aps[:, 1::2, ::2, :] = np.logical_or(aps[:, 1::2, ::2, :], np.logical_not(aps[:, ::2, 1::2]))
                aps[:, 1::2, 1::2, :] = np.logical_or(aps[:, 1::2, 1::2, :], np.logical_not(aps[:, ::2, ::2]))
                aps[:, 1::2, 1::2, :] = np.logical_or(aps[:, 1::2, 1::2, :], np.logical_not(aps[:, ::2, 1::2]))
                aps[:, 1::2, 1::2, :] = np.logical_or(aps[:, 1::2, 1::2, :], np.logical_not(aps[:, 1::2, ::2]))

                if vertical_padding:
                    aps = aps[:, 0:-1, :, :]
                if horizontal_padding:
                    aps = aps[:, :, 0:-1, :]

                current_activation_patterns.append(aps)
        return current_activation_patterns

    input = network.input
    outputs = []
    layers_without_softmax = network.layers[0:-1]
    for layer_idx in range(len(layers_without_softmax)):
        layer = layers_without_softmax[layer_idx]
        layer_type = get_layer_type(layer)
        if layer_type in ['leaky', 're', 'activation']:
            outputs.append(layer.output)
        elif layer_type == 'max':
            outputs.append(layers_without_softmax[layer_idx - 1].output)
            outputs.append(layer.output)

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    predictor = K.function([input], outputs)
    predicted_data = predictor([batch])
    activation_patterns = turn_data_into_activation_patterns(predicted_data)
    K.clear_session()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return activation_patterns


def get_epoch_numbers(arg):
    if arg.epochs == 'all':
        return ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
                '17', '18', '19', '20', '22', '24', '26', '28', '30', '32', '34', '36', '38', '40', '42', '44', '46',
                '48', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95', '100', '105', '110', '115', '120',
                '125', '130', '135', '140', '145', '150', '155', '160', '165', '170', '175', '180', '185', '190', '195',
                '200']
    elif arg.epochs == 'special':
        return ['30']
    else:
        return [None]


def get_associated_training_points(arg, network, x_test, x_train):
    pos_terms = np.vstack(load_tropical_function(arg, network, 'training', None))
    associated_training_points = []
    for subgroup in x_test:
        subgroup = prepare_data_for_tropical_function(subgroup)
        indices = np.argmax(np.dot(pos_terms, subgroup), axis=0)
        associated_training_points.append(x_train[indices])
    return associated_training_points


def get_no_data_points_per_label(grouped_data):
    return list(map(lambda y: sum(map(lambda x: x.shape[0], y)), grouped_data))


def get_no_data_subgroups_per_data_group(activation_patterns):
    return list(map(lambda x: len(x), activation_patterns))


def get_tropical_function_directory(arg, folder_name, data_type, epoch_number):
    save_dir = get_saving_directory(arg)
    func_dir = create_directory(save_dir, data_type.capitalize())
    if arg.epochs == 'all' or arg.epochs == 'special':
        func_dir = create_directory(func_dir, 'all_epochs', 'epoch_' + str(epoch_number))
    if arg.extract_all_dimensions:
        all_dimensions_string = '_all_dimensions'
    else:
        all_dimensions_string = ''
    func_dir = create_directory(func_dir, folder_name + all_dimensions_string)
    return func_dir


def flatten_and_stack(true_labels, network_labels, bias, B, batch_idx):
    B = np.reshape(B, newshape=[B.shape[0], -1])
    true_labels = np.expand_dims(true_labels[batch_idx], axis=1)
    network_labels = np.expand_dims(network_labels[batch_idx], axis=1)
    result = np.hstack([true_labels, network_labels, bias, B])
    return result


def prepare_data_for_tropical_function(data):
    return np.reshape(data, newshape=[data.shape[0], -1]).transpose()


def normalize(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def compute_similarity(terms_0, terms_1,epoch_number):
    def compute_1_1_angles(A, B):
        A = normalize(A)
        B = normalize(B)
        # clipping because there may be values slightly above 1/below -1
        return np.arccos(np.clip(np.einsum('ij,ij->i', A, B), -1, 1))

    def compute_1_1_correlations(A, B):
        A -= np.mean(A, axis=0, keepdims=True)
        B -= np.mean(B, axis=0, keepdims=True)
        A_B = np.sum(A * B, axis=0)
        A_A = np.sum(A * A, axis=0)
        B_B = np.sum(B * B, axis=0)
        divisor = np.sqrt(A_A) * np.sqrt(B_B)
        correlation = np.divide(A_B, divisor, out=np.ones_like(A_B), where=(divisor != 0))
        return correlation

    def compute_distances(A, B):
        return np.linalg.norm(A - B, axis=1)

    angles = compute_1_1_angles(terms_0, terms_1)
    correlations = compute_1_1_correlations(terms_0, terms_1)
    if epoch_number == '00':
        correlations[0] = 0
    distances = compute_distances(terms_0, terms_1)
    return angles, correlations, distances


def shift_array(array, v_shift, h_shift, filling='zeros'):
    shifted_array = np.zeros_like(array)
    if v_shift > 0:  # down shift
        shifted_array[:, v_shift:, :, :] = array[:, 0:(-v_shift), :, :]
        if filling == 'nearest':
            shifted_array[:, 0:v_shift, :, :] = array[:, 0:1, :, :]
        else:
            shifted_array[:, 0:v_shift, :, :] = 0
    elif v_shift < 0:  # up shift
        shifted_array[:, 0:v_shift, :, :] = array[:, (-v_shift):, :, :]
        if filling == 'nearest':
            shifted_array[:, v_shift:, :, :] = array[:, -1:, :, :]
        else:
            shifted_array[:, v_shift:, :, :] = 0
    else:
        shifted_array[:, :, :, :] = array[:, :, :, :]
    if h_shift > 0:  # right shift
        shifted_array[:, :, h_shift:, :] = shifted_array[:, :, 0:(-h_shift), :]
        if filling == 'nearest':
            shifted_array[:, :, 0:h_shift, :] = shifted_array[:, :, 0:1, :]
        else:
            shifted_array[:, :, 0:h_shift, :] = 0
    elif h_shift < 0:  # left shift
        shifted_array[:, :, 0:h_shift, :] = shifted_array[:, :, (-h_shift):, :]
        if filling == 'nearest':
            shifted_array[:, :, h_shift:, :] = shifted_array[:, :, -1:, :]
        else:
            shifted_array[:, :, h_shift:, :] = 0
    else:
        shifted_array[:, :, :, :] = shifted_array[:, :, :, :]
    return shifted_array


def shift_tropical_function(terms, v_shift, h_shift, filling='zeros'):
    no_labels = len(terms)
    for i in range(no_labels):
        bias = terms[i][:, 0:1]
        main = terms[i][:, 1:]
    logger.info("Shifts in [-6, -3, 0, 3, 6]^2")
    function_path = get_function_path(arg, last_layer_index, transformation_path)
    out = [None] * no_labels
    for k in range(no_labels):
        pos_terms_k = np.load(function_path + 'pos_label_' + str(k) + '.npy')
        k_bias = pos_terms_k[:, 0:1]
        k_main = pos_terms_k[:, 1:]
        k_main = k_main.reshape([-1, 32, 32, 3])
        shifts = [-6, -3, 0, 3, 6]
        for v_shift in shifts:
            for h_shift in shifts:
                if v_shift != 0 or h_shift != 0:
                    shifted_terms = shift_array(k_main, v_shift, h_shift, shift_type)
                    shifted_terms = shifted_terms.reshape([-1, 3072])
                    shifted_terms = np.hstack([k_bias, shifted_terms])
                    pos_terms_k = np.vstack([pos_terms_k, shifted_terms])
        out[k] = np.max(np.dot(pos_terms_k, current_data), axis=0)
    tropical_augmented_test_labels = np.argmax(out, axis=0)
    network_agreement_augmented = sum(tropical_augmented_test_labels == network_labels) / len(network_labels)
