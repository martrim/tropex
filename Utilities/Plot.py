from matplotlib import pyplot as plt
import numpy as np
import os
from Utilities.Saver import create_directory, get_file_name, get_saving_directory


def plot_loss_and_accuracy(history, arg):
    # plot loss
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    l1, = axs[0].plot(history['loss'], color='blue', label='train')
    l2, = axs[0].plot(history['val_loss'], color='orange', label='test')
    axs[0].set_title('Cross Entropy Loss', fontsize=10)
    axs[0].legend((l1, l2), ('train', 'test'), loc='upper right')
    fig.suptitle(os.path.join('Activation: ' + arg.activation_function, 'OH Factor: ' + str(arg.overheating_factor),
                              '\n', 'Maximum Temperature: ' + str(arg.maximum_temperature),
                              'Reset Temperature: ' + str(arg.reset_temperature)), fontsize=12, fontweight='bold')
    # plot accuracy
    l1, = axs[1].plot(history['accuracy'], color='blue', label='train')
    l2, = axs[1].plot(history['val_accuracy'], color='orange', label='test')
    axs[1].set_title('Classification Accuracy', fontsize=10)
    axs[1].legend((l1, l2), ('train', 'test'), loc='lower right')
    # Saving
    save_dir = get_saving_directory(arg)
    accuracy_and_loss_directory = create_directory(save_dir, 'Plots', 'Accuracy_and_Loss')
    plot_name = get_file_name(arg, '.png')
    file_name = os.path.join(accuracy_and_loss_directory, plot_name)
    plt.savefig(file_name)
    plt.close()


def plot_ECEs_and_temperatures(arg):
    save_dir = get_saving_directory(arg)
    array_name = get_file_name(arg, '.npy')
    array_directory = os.path.join(save_dir, 'Arrays')
    path = os.path.join(array_directory, 'Temperatures', array_name)
    if os.path.isfile(path):
        plot_dir = create_directory(save_dir, 'Plots')
        plot_name = get_file_name(arg, '.png')
        # Temperatures
        temperatures = np.load(path)
        plt.plot(temperatures)
        plt.title('Temperatures')
        temperature_dir = create_directory(plot_dir, 'Temperatures')
        file_name = os.path.join(temperature_dir, plot_name)
        plt.savefig(file_name)
        plt.close()
        # ECEs
        before_scaling_path = os.path.join(array_directory, 'ECEs_before_scaling', array_name)
        ECEs_before_scaling = np.load(before_scaling_path)
        after_scaling_path = os.path.join(array_directory, 'ECEs_after_scaling', array_name)
        ECEs_after_scaling = np.load(after_scaling_path)
        plt.plot(ECEs_before_scaling, color='blue', label='before scaling')
        plt.plot(ECEs_after_scaling, color='orange', label='after scaling')
        plt.title('ECEs')
        ECE_dir = create_directory(plot_dir, 'ECEs')
        file_name = os.path.join(ECE_dir, plot_name)
        plt.savefig(file_name)
        plt.close()
