import csv
import numpy as np
import os
import pickle
import socket


def get_network_location(arg, epoch_number):
    save_dir = get_saving_directory(arg)
    network_dir = create_directory(save_dir, 'Models')
    if arg.epochs == 'all' or arg.epochs == 'special':
        network_dir = create_directory(network_dir, 'all_epochs')
    network_name = get_file_name(arg, '.h5', epoch_number)
    network_path = os.path.join(network_dir, network_name)
    return network_path


def get_file_name(arg, ending, epoch_number=None):
    if arg.temperature_scaling:
        file_name = 'oh_factor_' + str(arg.overheating_factor) + '__max_temp_' + str(arg.maximum_temperature) + \
                    '__reset_temp_' + str(arg.reset_temperature) + '{epoch:02d}' + ending
    else:
        file_name = 'data_augmentation_' + str(arg.data_augmentation) + '__early_stopping_' + str(arg.early_stopping) \
                    + '__dropout_' + str(arg.dropout) + '__weight_decay_' + str(arg.weight_decay)

        if (epoch_number is None) or (arg.epochs != 'all' and arg.epochs != 'special'):
            file_name += ending
        else:
            if epoch_number == 'variable':
                file_name += '_epoch_{epoch:02d}' + ending
            else:
                file_name += '_epoch_' + str(epoch_number) + ending
    return file_name


def create_directory(*args):
    directory = '/'.join(args)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


def get_saving_directory(arg):
    if socket.gethostname() == 'dechter' or socket.gethostname() == 'maclaurin':
        basic_path = '/home/martin/'
    elif socket.gethostname() == 'lebesgue':
        basic_path = '/media/hpetzka/C4BADFE8BADFD550/'
    else:
        basic_path = ''
    basic_path = os.path.join(basic_path, os.getcwd().split('/')[-1])
    if arg.temperature_scaling:
        save_dir = create_directory(basic_path, 'DEBUGGING_Results', arg.network_type_coarse, arg.activation_function)
    else:
        save_dir = create_directory(basic_path, 'DEBUGGING_Results', arg.data_set, arg.network_type_coarse, arg.network_type_fine,
                                arg.network_number)
    return save_dir


def load_history(arg):
    save_dir = get_saving_directory(arg)
    history_dir = os.path.join(save_dir, 'Histories')
    file_name = get_file_name(arg, '')
    history_path = os.path.join(history_dir, file_name)
    with open(history_path, 'rb') as history_file:
        history = pickle.load(history_file)
    return history


def save(history, arg):
    save_dir = get_saving_directory(arg)

    # Saving the history
    history_dir = create_directory(save_dir, 'Histories')
    file_name = get_file_name(arg, '')
    history_path = os.path.join(history_dir, file_name)
    with open(history_path, 'wb') as history_file:
        pickle.dump(history, history_file)

    # Saving the documentation for the Google Sheet
    documentation_directory = create_directory(save_dir, 'Documentation')
    file_name = get_file_name(arg, '.csv')
    accuracy_and_loss_path = os.path.join(documentation_directory, file_name)
    val_accuracy = history['val_accuracy']
    no_epochs = len(val_accuracy)
    max_accuracy = np.max(val_accuracy)
    argmax_accuracy = np.argmax(val_accuracy) + 1
    final_accuracy = val_accuracy[-1]
    val_loss = history['val_loss']
    min_loss = np.min(val_loss)
    argmin_loss = np.argmin(val_loss) + 1
    final_loss = val_loss[-1]
    if arg.temperature_scaling:
        documentation = [arg.activation_function, arg.data_augmentation, arg.gradient_tape, arg.lr_schedule,
                         arg.overheating_factor, arg.maximum_temperature, arg.reset_temperature,
                         arg.initial_temperature, arg.batch_size, no_epochs,
                         max_accuracy, argmax_accuracy, final_accuracy, min_loss, argmin_loss, final_loss]
    else:
        documentation = [arg.activation_function, arg.data_augmentation, arg.early_stopping, arg.dropout,
                         arg.weight_decay, arg.gradient_tape, arg.lr_schedule, arg.batch_size, no_epochs,
                         max_accuracy, argmax_accuracy, final_accuracy, min_loss, argmin_loss, final_loss]
    documentation = [list(map(lambda x: str(x), documentation))]
    with open(accuracy_and_loss_path, 'w') as csv_file:
        wr = csv.writer(csv_file, dialect='excel')
        wr.writerows(documentation)
