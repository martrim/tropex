import numpy as np
# from Cleverhans.basic_iterative_method import basic_iterative_method
# from Cleverhans.fast_gradient_method import fast_gradient_method
# from Cleverhans.madry_et_al import madry_et_al
# from Cleverhans.momentum_iterative_method import momentum_iterative_method
# from Cleverhans.projected_gradient_descent import projected_gradient_descent
# from Cleverhans.spsa import spsa


def generate_adversarial_data(network, x_train, method='fast_gradient'):
    batch_size = 5000
    no_batches = int(x_train.shape[0] / batch_size)
    adv_x = [None] * no_batches
    if method == 'fast_gradient':
        for i in range(no_batches):
            adv_x[i] = fast_gradient_method(model_fn=network, x=x_train[5000 * i:5000 * (i + 1)], eps=0.25, norm=1)
    elif method == 'basic_iterative':
        for i in range(no_batches):
            adv_x[i] = basic_iterative_method(network, x_train[5000 * i:5000 * (i + 1)], 0.25, 1)
    elif method == 'madry_et_al':
        for i in range(no_batches):
            adv_x[i] = madry_et_al(network, x_train[5000 * i:5000 * (i + 1)], 0.25, 1)
    elif method == 'momentum_iterative':
        for i in range(no_batches):
            adv_x[i] = momentum_iterative_method(model_fn=network, x=x_train[5000 * i:5000 * (i + 1)])
    elif method == 'projected_gradient_descent':
        for i in range(no_batches):
            adv_x[i] = projected_gradient_descent(network, x_train[5000 * i:5000 * (i + 1)], 0.25, 1)
    elif method == 'spsa':
        for i in range(no_batches):
            adv_x[i] = spsa(network, x_train[5000 * i:5000 * (i + 1)], 0.25, 1)
    adv_x = np.vstack(adv_x)
    return adv_x


def subtract_pixel_mean(x_train, x_test):
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    return x_train, x_test


# load train and test dataset
def load_data(arg, data_type='all'):
    def scale_pixels(x_train, x_test):
        return x_train.astype('float32') / 255, x_test.astype('float32') / 255

    if arg.data_set == 'CIFAR10':
        if data_type == 'random':
            return np.random.rand(10**4, 32, 32, 3), np.random.randint(0, 10, size=[10**4])
        elif data_type == 'random_100':
            return 200*np.random.rand(10**4, 32, 32, 3)-100, np.random.randint(0, 10, size=[10**4])
        elif data_type == 'random_1000':
            return 2000*np.random.rand(10**4, 32, 32, 3)-1000, np.random.randint(0, 10, size=[10**4])
        elif data_type == 'random_10000':
            return 20000*np.random.rand(10**4, 32, 32, 3)-10000, np.random.randint(0, 10, size=[10**4])

        from tensorflow.keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif arg.data_set == 'MNIST':
        if data_type == 'random':
            return np.random.rand(10**4, 28, 28, 1), np.random.randint(0, 10, size=[10**4])
        elif data_type == 'random_100':
            return 200*np.random.rand(10**4, 28, 28, 1)-100, np.random.randint(0, 10, size=[10**4])
        elif data_type == 'random_1000':
            return 2000 * np.random.rand(10 ** 4, 28, 28, 3) - 1000, np.random.randint(0, 10, size=[10 ** 4])
        elif data_type == 'random_10000':
            return 20000 * np.random.rand(10 ** 4, 28, 28, 3) - 10000, np.random.randint(0, 10, size=[10 ** 4])


        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
    elif arg.data_set == 'Fashion MNIST':
        if data_type == 'random':
            return np.random.rand(10**4, 28, 28, 1), np.random.randint(0, 10, size=[10**4])
        elif data_type == 'random_100':
            return 200*np.random.rand(10**4, 28, 28, 1)-100, np.random.randint(0, 10, size=[10**4])
        elif data_type == 'random_1000':
            return 2000*np.random.rand(10**4, 28, 28, 3)-1000, np.random.randint(0, 10, size=[10**4])
        elif data_type == 'random_10000':
            return 20000*np.random.rand(10**4, 28, 28, 3)-10000, np.random.randint(0, 10, size=[10**4])

        from tensorflow.keras.datasets import fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
    else:
        return None

    x_train, x_test = scale_pixels(x_train, x_test)
    y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)

    if arg.network_type_coarse == 'ResNet':
        x_train, x_test= subtract_pixel_mean(x_train, x_test)

    if data_type == 'all':
        return x_train, y_train, x_test, y_test
    elif data_type == 'training':
        return x_train, y_train
    elif data_type == 'test':
        return x_test, y_test