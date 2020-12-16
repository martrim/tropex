import tensorflow as tf
from tensorflow.keras.layers import ReLU, LeakyReLU, Softmax
from Utilities.Custom_Activations import split_relu
from Utilities.Saver import get_network_location
from tensorflow.keras.optimizers import Adam, SGD
from Utilities.Callbacks import get_scheduler

def load_network(arg, epoch_number):
    learning_rate = get_scheduler(arg)(epoch=0)
    if arg.network_type_coarse == 'ResNet' or arg.data_set == 'MNIST':
        opt = Adam(lr=learning_rate)
    else:
        opt = SGD(lr=learning_rate, momentum=0.9)

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr

        return lr

    network_location = get_network_location(arg, epoch_number)
    custom_objects = {"split_relu": split_relu, "LeakyReLU": LeakyReLU(), "ReLU": ReLU(), "Softmax": Softmax(), "lr":get_lr_metric(opt)}
    network = tf.keras.models.load_model(network_location, custom_objects=custom_objects)
    return network