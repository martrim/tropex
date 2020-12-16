import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from Utilities.Saver import get_network_location


def get_scheduler(arg):
    def general_schedule(epoch):
        S = [50, 100, 150]
        initial_lrate = 0.01
        drop_rate = 1e-1
        if epoch < S[0]:
            lrate = initial_lrate
        elif epoch < S[1]:
            lrate = initial_lrate * drop_rate
        elif epoch < S[2]:
            lrate = initial_lrate * (drop_rate ** 2)
        else:
            lrate = initial_lrate * (drop_rate ** 3)
        print('learning rate:')
        print(lrate)
        return lrate

    def ResNet_schedule(epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        return lr

    def MNIST_schedule(epoch):
        lr = 1e-4
        if epoch > 60:
            lr *= 1e-6
        elif epoch > 100:
            lr *= 1e-5
        elif epoch > 40:
            lr *= 1e-4
        elif epoch > 30:
            lr *= 1e-3
        elif epoch > 20:
            lr *= 1e-2
        elif epoch > 10:
            lr *= 1e-1
        return lr

    if arg.data_set == 'MNIST' or arg.data_set == 'Fashion MNIST':
        return MNIST_schedule
    if arg.network_type_coarse in ['-', 'AllCNN', 'FCN', 'VGG']:
        return general_schedule
    elif arg.network_type_coarse == 'ResNet':
        return ResNet_schedule


def get_callbacks(x_test, y_test, arg):
    callbacks = []
    # Checkpoint
    network_path = get_network_location(arg, epoch_number='variable')
    if arg.epochs == 'all' or arg.epochs == 'special':
        save_best_only = False
    else:
        save_best_only = True
    checkpoint = tf.keras.callbacks.ModelCheckpoint(network_path, monitor='val_accuracy', verbose=0,
                                                    save_best_only=save_best_only,
                                                    save_weights_only=False, mode='auto', save_freq='epoch')
    callbacks.append(checkpoint)
    # Temperature Scaling
    if arg.temperature_scaling:
        from rhTemperature_Functions import TemperatureScaleCallback
        true_labels = np.argmax(y_test, axis=1)
        callbacks.append(TemperatureScaleCallback(x_test, y_test, true_labels, arg))
    # Early Stopping
    if arg.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=0,
                                       mode='auto', baseline=None))
    # Learning Rate Schedule
    if arg.lr_schedule:
        callbacks.append(LearningRateScheduler(get_scheduler(arg)))

    if arg.network_type_coarse == 'ResNet':
        callbacks.append(ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6))
    # Tensorboard
    if arg.tensorboard:
        tensorboard_dir = os.path.join(save_dir, 'tensorboard')
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1))
    return callbacks
