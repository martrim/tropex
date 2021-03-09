import numpy as np
import tensorflow as tf
from Utilities.Tropical_Helper_Functions import flatten_and_stack, get_tropical_function_directory, \
    get_activation_patterns, get_layer_type, get_epoch_numbers, get_tropical_filename_ending, get_batch_data, \
    get_folder_name
from Utilities.Custom_Settings import apply_resnet_settings, configure_gpu
from Utilities.Logger import *
from Utilities.Network_Loader import load_network
from Utilities.Parser import parse_arguments

start_time = print_start()

# To raise an exception on runtime warnings, used for debugging.
np.seterr(all='raise')

# Load the arguments
arg = parse_arguments()
if arg.network_type_coarse == 'ResNet':
    arg = apply_resnet_settings(arg)

# Configure the GPU for Tensorflow
configure_gpu(arg)


def transform_batch(batch_idx, sign):
    def save(batch_idx, layer_idx, B, bias):
        folder_name = get_folder_name(network, layer_idx)
        save_dir = get_tropical_function_directory(arg, folder_name, arg.data_type, epoch_number)
        file_name_ending = get_tropical_filename_ending(batch_idx)

        B = flatten_and_stack(true_labels, network_labels, bias, B, batch_idx)
        np.save(os.path.join(save_dir, sign + file_name_ending), B)

    def add(B_0, bias_0, B_max_0, B_1, bias_1, B_max_1):
        B = B_0 + B_1
        bias = bias_0 + bias_1
        B_max = B_max_0 + B_max_1
        return B, bias, B_max

    def avpool(B, B_max, pool_size):
        pool_divisor = pool_size[0] * pool_size[1]
        B_max = np.repeat(np.repeat(B_max, repeats=pool_size[0], axis=1), repeats=pool_size[1], axis=2)
        B_max /= pool_divisor
        B = np.repeat(np.repeat(B, repeats=pool_size[0], axis=1), repeats=pool_size[1], axis=2)
        B /= pool_divisor
        return B, B_max

    def batch(B, bias, B_max):
        def merge_layers(scale, bias, B, bias_merged):
            bias_merged += np.dot(B.reshape([B.shape[0], -1]), bias[:, np.newaxis])
            B = B * scale
            return B, bias_merged

        gamma, beta, mean, var = layer.get_weights()
        gamma = gamma[np.newaxis, np.newaxis, np.newaxis, :]
        beta = beta[np.newaxis, np.newaxis, np.newaxis, :]
        mean = mean[np.newaxis, np.newaxis, np.newaxis, :]
        var = var[np.newaxis, np.newaxis, np.newaxis, :]
        eps = layer.epsilon
        # data_after_batchnorm = ((data_before_batchnorm - mean)/np.sqrt(var+epsilon))*gamma + beta
        scale = gamma / np.sqrt(var + eps)
        translation = np.squeeze(-mean * scale + beta)
        B_max = B_max * np.abs(scale)

        _, input_height, input_width, input_channels = layer.input_shape
        new_bias = np.tile(translation, input_height * input_width)

        B, bias = merge_layers(scale, new_bias, B, bias)
        return B, bias, B_max

    def copy(B, bias, B_max):
        B_0 = B
        bias_0 = bias
        B_max_0 = B_max
        B_1 = np.copy(B)
        bias_1 = np.copy(bias)
        B_max_1 = np.copy(B_max)
        return B_0, bias_0, B_max_0, B_1, bias_1, B_max_1

    def convolution(B, bias, B_max):
        def merge_layers(current_layer, B, bias):
            filt, layer_bias = current_layer.get_weights()
            filt = filt.astype('float64')
            layer_bias = layer_bias.astype('float64')
            new_output_shape = (B.shape[0],) + current_layer.input_shape[1:]
            strides = (1,) + current_layer.strides + (1,)
            B_new = tf.nn.conv2d_transpose(B, filt, output_shape=new_output_shape,
                                           strides=strides, padding='SAME')
            bias_repeated = np.tile(layer_bias, output_height * output_width)
            bias_new = np.dot(B.reshape([B.shape[0], -1]), bias_repeated[:, np.newaxis]) + bias
            return B_new, bias_new

        output_shape = layer.output_shape[1:]
        output_height, output_width, output_channels = output_shape
        filt, layer_bias = layer.get_weights()
        filt = filt.astype('float64')

        filt_neg = (-filt) * (filt < 0)
        filt_pos = filt * (filt > 0)
        new_output_shape = (1,) + layer.input_shape[1:]
        strides = (1,) + layer.strides + (1,)
        K = tf.nn.conv2d_transpose(B_max, filt_neg, output_shape=new_output_shape,
                                   strides=strides, padding='SAME')
        B_max = tf.nn.conv2d_transpose(B_max, filt_pos, output_shape=new_output_shape,
                                       strides=strides, padding='SAME')
        B_max += K

        B, bias = merge_layers(layer, B, bias)
        B += K
        return B, bias, B_max

    def dense(B, bias, B_max):
        def merge_layers(W, layer_bias, B, bias):
            bias += np.dot(B, layer_bias)
            B = np.dot(B, W)
            B += K
            return B, bias

        W, layer_bias = layer.get_weights()
        W = W.astype('float64')
        W = W.transpose()
        layer_bias = layer_bias.astype('float64')
        layer_bias = layer_bias[:, np.newaxis]
        K = np.dot(B_max, (-W) * (W < 0))
        B_max = K + np.dot(B_max, W * (W > 0))

        B, bias = merge_layers(W, layer_bias, B, bias)

        return B, bias, B_max

    def leaky(B, alpha):
        B[activation_patterns[counter]] *= alpha
        return B

    def maxpool(B, B_max, input_shape):
        def repeat(array, input_shape):
            array = np.repeat(np.repeat(array, repeats=2, axis=1), repeats=2, axis=2)
            array = array[:, 0:input_shape[1], 0:input_shape[2], :]
            return array

        B_max = repeat(B_max, input_shape)
        B = repeat(B, input_shape)
        B[activation_patterns[counter]] = 0
        return B, B_max

    def relu(B):
        B[activation_patterns[counter]] = 0
        return B

    if sign == 'pos':
        B = A_plus[network_labels[batch_idx]]
        B_max = np.max(np.vstack([A_plus, A_minus]), axis=0)
    elif sign == 'neg':
        B = np.repeat(A_minus, network_labels[batch_idx].shape[0], axis=0)
        B_max = np.max(np.vstack([A_plus, A_minus]), axis=0)
    elif sign == 'linear':
        B = W[network_labels[batch_idx]]
        B_max = np.zeros_like(A_plus[0])
    else:
        return

    bias = B[:, 0:1]
    B_max = B_max[1:]
    B = B[:, 1:]

    if last_layer_type == 'global':
        B = B.reshape([-1, h, w, c])
        B_max = B_max.reshape([-1, h, w, c])

    if arg.save_intermediate:
        save(batch_idx, len(network.layers)-1, B, bias)

    counter = 0
    input_names = []
    for layer_idx, layer in reversed(list(enumerate(network.layers[0:-2]))):
        layer_type = get_layer_type(layer)
        if len(input_names) > 1:
            if input_names[0] == layer.name:
                B, bias, B_max = B_0, bias_0, B_max_0
            elif input_names[1] == layer.name:
                B, bias, B_max = B_1, bias_1, B_max_1
        if layer_type == 'add':
            inputs = layer.input
            input_names = [input.name.split('/')[0] for input in inputs]
            B_0, bias_0, B_max_0, B_1, bias_1, B_max_1 = copy(B, bias, B_max)
        elif layer_type == 'average':
            B, B_max = avpool(B, B_max, layer.pool_size)
        elif layer_type == 'batch':
            B, bias, B_max = batch(B, bias, B_max)
        elif layer_type == 'conv2d':
            B, bias, B_max = convolution(B, bias, B_max)
            B = B.numpy()
        elif layer_type == 'dense':
            B, bias, B_max = dense(B, bias, B_max)
        elif layer_type == 'dropout':
            pass
        elif layer_type == 'flatten':
            B = B.reshape((-1,) + layer.input_shape[1:])
            B_max = B_max.reshape((-1,) + layer.input_shape[1:])
        elif layer_type == 'leaky':
            B = leaky(B, layer.alpha)
            counter += 1
        elif layer_type == 'max':
            B, B_max = maxpool(B, B_max, layer.input_shape)
            counter += 1
        elif layer_type == 're' or layer_type == 'activation':
            B = relu(B)
            counter += 1

        if len(input_names) > 1:
            if input_names[0] == layer.name:
                B_0, bias_0, B_max_0 = B, bias, B_max
                input_names[0] = layer.input.name.split('/')[0]
            elif input_names[1] == layer.name:
                B_1, bias_1, B_max_1 = B, bias, B_max
                input_names[1] = layer.input.name.split('/')[0]

            if input_names[0] == input_names[1]:
                B, bias, B_max = add(B_0, bias_0, B_max_0, B_1, bias_1, B_max_1)
                input_names = []

        if arg.save_intermediate and layer_type in saving_types:
            save(batch_idx, layer_idx, B, bias)
        logger.info('Done with merge ' + str(layer_idx) + ' of ' + str(last_layer_index))

    save(batch_idx, layer_idx, B, bias)
    logger.info('Done with batch ' + str(batch_idx + 1) + ' of ' + str(no_batches))


logger = get_logger(arg)
epoch_numbers = get_epoch_numbers(arg)

for epoch_number in epoch_numbers:
    print('Epoch number: ' + str(epoch_number))
    logger.info('Epoch number: ' + str(epoch_number))
    network = load_network(arg, epoch_number)
    # subtract 2 because we skip the final activation and indexing starts from 0 and not from 1
    last_layer_index = len(network.layers) - 2
    no_labels = network.layers[-1].output_shape[1]
    data_batches, true_labels, network_labels = get_batch_data(arg, network)
    no_batches = len(data_batches)

    for batch_idx in range(no_batches):
        print('No of data in batch ' + str(batch_idx + 1) + ': ' + str(data_batches[batch_idx].shape[0]))

    saving_types = ['conv2d', 'dense', 'global', 'max']

    last_layer_type = get_layer_type(network.layers[last_layer_index])
    if last_layer_type == 'dense':
        weights, bias = network.layers[last_layer_index].get_weights()
        weights = weights.astype('float64')
        bias = bias.astype('float64')
        W = np.vstack([bias, weights]).transpose()
    elif last_layer_type == 'global':
        h, w, c = network.layers[last_layer_index].input_shape[1:]
        W = np.zeros([c, h * w, c])
        for gamma in range(c):
            W[gamma, :, gamma] = 1 / (h * w)
        W = W.reshape(c, -1)
        W = np.hstack([np.zeros([c, 1]), W])
    else:
        raise Exception('Incorrect type of the last layer.')

    A_minus = np.expand_dims(np.sum((-W) * (W < 0), axis=0), axis=0)
    A_plus = W + A_minus

    for batch_idx in range(no_batches):
        print('Batch ' + str(batch_idx+1) + ' of ' + str(no_batches))
        activation_patterns = get_activation_patterns(arg, network, batch=data_batches[batch_idx])
        if (arg.extraction_type == 'pos_and_neg') or (arg.extraction_type == 'pos'):
            transform_batch(batch_idx, 'pos')
        if (arg.extraction_type == 'pos_and_neg') or (arg.extraction_type == 'neg'):
            transform_batch(batch_idx, 'neg')
        if arg.extraction_type == 'linear':
            transform_batch(batch_idx, 'linear')

for handler in logger.handlers:
    handler.close()
    logger.removeHandler(handler)

print_end(start_time)
