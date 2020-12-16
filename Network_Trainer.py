import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Utilities.Callbacks import get_callbacks, get_scheduler
from Utilities.Custom_Settings import apply_resnet_settings
from Utilities.Logger import print_start, print_end
from Utilities.Networks import define_model
from Utilities.Data_Loader import load_data
from Utilities.Parser import parse_arguments
from Utilities.Plot import plot_ECEs_and_temperatures, plot_loss_and_accuracy
from Utilities.Saver import get_network_location, save

start_time = print_start()

arg = parse_arguments()
if arg.network_type_coarse == 'ResNet':
    arg = apply_resnet_settings(arg)

# Set available GPU
os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu

# Show training bar only if run in the IDE
if "PYCHARM_HOSTED" in os.environ:
    training_verbosity = 1
else:
    training_verbosity = 2


def main(arg):
    x_train, y_train, x_test, y_test = load_data(arg, data_type='all')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    callbacks = get_callbacks(x_test, y_test, arg)
    model = define_model(arg)
    learning_rate = get_scheduler(arg)(epoch=0)
    if arg.network_type_coarse == 'ResNet' or arg.data_set == 'MNIST':
        opt = Adam(lr=learning_rate)
    else:
        opt = SGD(lr=learning_rate, momentum=0.9)

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr

    lr_metric = get_lr_metric(opt)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', lr_metric], run_eagerly=arg.train_eagerly)
    if arg.epochs == 'all' or arg.epochs == 'special':
        network_path = get_network_location(arg, '00')
        model.save(network_path)
    if arg.data_augmentation:
        num_samples = x_train.shape[0]
        if arg.network_type_coarse == 'ResNet':
            data_gen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                                         featurewise_std_normalization=False, samplewise_std_normalization=False,
                                         zca_whitening=False, zca_epsilon=1e-06, rotation_range=0,
                                         width_shift_range=0.1, height_shift_range=0.1, shear_range=0., zoom_range=0.,
                                         channel_shift_range=0., fill_mode='nearest', cval=0., horizontal_flip=True,
                                         vertical_flip=False, rescale=None, preprocessing_function=None,
                                         data_format=None, validation_split=0.0)
            data_gen.fit(x_train)
        else:
            data_gen = ImageDataGenerator(width_shift_range=int(6), height_shift_range=int(6), horizontal_flip=True)
        it = data_gen.flow(x_train, y_train)
        history = model.fit(it, steps_per_epoch=np.ceil(num_samples / arg.batch_size),
                            epochs=arg.no_epochs, validation_data=(x_test, y_test), callbacks=callbacks,
                            verbose=training_verbosity)
    else:
        history = model.fit(x_train, y_train, batch_size=arg.batch_size,
                            epochs=arg.no_epochs, validation_data=(x_test, y_test), callbacks=callbacks,
                            verbose=training_verbosity)
    history = history.history
    save(history, arg)
    plot_loss_and_accuracy(history, arg)
    if arg.temperature_scaling:
        plot_ECEs_and_temperatures(arg)


main(arg)

print_end(start_time)