import os
import tensorflow as tf


def apply_resnet_settings(arg):
    arg.batch_size = 32 # orig paper trained all networks with batch_size=128
    arg.no_epochs = 200
    arg.data_augmentation = True
    arg.lr_schedule = True
    return arg


def configure_gpu(arg):
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)