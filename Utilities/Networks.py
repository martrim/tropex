from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Input, add
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Activation, ReLU, LeakyReLU, Softmax
from Utilities.Custom_Activations import split_relu
from Utilities.Sequential_with_Gradient_Tape import Sequential_with_Gradient_Tape

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, kernel_initializer='he_normal')(y)
    outputs = Activation('softmax')(outputs)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def define_model(arg):
    if arg.data_set == 'CIFAR10':
        input_shape = (32, 32, 3)
    else:
        input_shape = (28, 28, 1)
    
    model_type_fine = arg.network_type_fine
    dropout = arg.dropout
    if arg.network_type_coarse == 'AllCNN' or model_type_fine == 'Wide':
        regularizer = True
    else:
        regularizer = arg.weight_decay

    # Get the activation function
    if arg.activation_function == 'relu':
        activation = ReLU()
        channel_reduction = 1.
    elif arg.activation_function == 'leaky_relu':
        activation = LeakyReLU()
        channel_reduction = 1.
    elif arg.activation_function == 'split_relu':
        activation = split_relu
        channel_reduction = 3. / 4.
    else:
        activation = ReLU()
        channel_reduction = 1.

    # General
    def add_last_layers(no_layers=2):
        model.add(Flatten())
        if no_layers == 3:
            if regularizer:
                model.add(Dense(1024, kernel_initializer='he_uniform', kernel_regularizer=l2(0.001)))
            else:
                model.add(Dense(1024, kernel_initializer='he_uniform'))
            model.add(Activation(activation))
        if no_layers >= 2:
            if regularizer:
                model.add(Dense(128, kernel_initializer='he_uniform', kernel_regularizer=l2(0.001)))
            else:
                model.add(Dense(128, kernel_initializer='he_uniform'))
            model.add(Activation(activation))
        if regularizer:
            model.add(Dense(10, kernel_initializer='he_uniform', kernel_regularizer=l2(0.001)))
        else:
            model.add(Dense(10, kernel_initializer='he_uniform'))
        model.add(Activation(Softmax()))

    # AllCNN
    def add_AllCNN_block(n_filters, filter_size=(3, 3), strides=(1, 1), input=False):
        if regularizer:
            if input:
                model.add(Conv2D(n_filters, filter_size, kernel_initializer='he_uniform', kernel_regularizer=l2(0.001),
                                 padding='same', strides=strides, input_shape=input_shape))
                model.add(Activation(activation))
            else:
                model.add(Conv2D(n_filters, filter_size, kernel_initializer='he_uniform', kernel_regularizer=l2(0.001),
                                 padding='same', strides=strides))
                model.add(Activation(activation))
        else:
            if input:
                model.add(Conv2D(n_filters, filter_size, kernel_initializer='he_uniform',
                                 padding='same', strides=strides, input_shape=input_shape))
                model.add(Activation(activation))
            else:
                model.add(Conv2D(n_filters, filter_size, kernel_initializer='he_uniform',
                                 padding='same', strides=strides))
                model.add(Activation(activation))
        model.add(Activation(activation))

    # FCN
    def add_FCN_block(no_nodes):
        if regularizer:
            model.add(Dense(no_nodes, kernel_regularizer=l2(0.001)))
        else:
            model.add(Dense(no_nodes))
        model.add(Activation(activation))

    # VGG
    def add_VGG_block(n_filters, filter_size=(3, 3), n_conv=2, input=False, maxpooling=True, avpooling=False,
                      batchnorm=False, strides=1):
        if input:
            if regularizer:
                model.add(Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform',
                                 input_shape=input_shape, kernel_regularizer=l2(0.001)))
                model.add(Activation(activation))
            else:
                model.add(Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform',
                                 input_shape=input_shape))
                model.add(Activation(activation))
            no_remaining_convolutions = n_conv - 1
        else:
            if regularizer:
                model.add(Conv2D(n_filters, filter_size, strides=strides, padding='same', kernel_initializer='he_uniform',
                                 kernel_regularizer=l2(0.001)))
                model.add(Activation(activation))
            else:
                model.add(Conv2D(n_filters, filter_size, strides=strides, padding='same', kernel_initializer='he_uniform'))
                model.add(Activation(activation))
            no_remaining_convolutions = n_conv
        for _ in range(no_remaining_convolutions):
            if regularizer:
                model.add(Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform',
                                 kernel_regularizer=l2(0.001)))
                model.add(activation)
            else:
                model.add(Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform'))
                model.add(Activation(activation))
            if batchnorm:
                model.add(BatchNormalization())
        if batchnorm:
            model.add(BatchNormalization())
        if strides > 1:
            avpooling = maxpooling = False
        if avpooling:
            maxpooling = False
            model.add(AveragePooling2D((2, 2)))
        if maxpooling:
            model.add(MaxPooling2D((2, 2), padding='same'))

    # ResNet
    def ResNet_block(input, n_filters, no_layers, first_layer=False):
        def residual_layer(input, n_filters, kernel_size=3, strides=1):
            output = Conv2D(n_filters, kernel_size=kernel_size, strides=strides, padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input)
            output = Activation('relu')(output)
            return output

        def residual_module(input, n_filters, downsample=False):
            if downsample:
                strides = 2
            else:
                strides = 1
            conv_1 = residual_layer(input, n_filters, strides=strides)
            conv_2 = residual_layer(conv_1, n_filters)
            if downsample:
                input = residual_layer(input, n_filters, kernel_size=1, strides=strides)
            output = add([conv_2, input])
            output = Activation('relu')(output)
            return output

        if first_layer:
            downsample = False
            input = residual_layer(input, n_filters, kernel_size=1)
        else:
            downsample = True
        intermediate_result = residual_module(input, n_filters, downsample)
        for i in range(no_layers - 1):
            intermediate_result = residual_module(intermediate_result, n_filters)
        return intermediate_result

    def dense_block_resnet(input, no_outputs):
        pooled = GlobalAveragePooling2D()(input)
        output = Dense(no_outputs, kernel_initializer='he_normal')(pooled)
        output = Activation(Softmax())(output)
        return output

    if arg.gradient_tape:
        model = Sequential_with_Gradient_Tape()
    else:
        model = Sequential()
    if arg.network_type_coarse == 'AllCNN':
        if model_type_fine == 'Standard':
            add_AllCNN_block(96 * channel_reduction, input=True)
            model.add(Activation(activation))
            model.add(Dropout(0.2))
            add_AllCNN_block(96 * channel_reduction)
            add_AllCNN_block(96 * channel_reduction, strides=(2, 2))
            model.add(Dropout(0.5))
            add_AllCNN_block(192 * channel_reduction)
            add_AllCNN_block(192 * channel_reduction)
            add_AllCNN_block(192 * channel_reduction, strides=(2, 2))
            model.add(Activation(activation))
            model.add(Dropout(0.5))
            add_AllCNN_block(192 * channel_reduction)
            add_AllCNN_block(192 * channel_reduction)
            add_AllCNN_block(10, filter_size=(1, 1))
            model.add(GlobalAveragePooling2D())
            model.add(Activation(Softmax()))
        elif model_type_fine == 'Narrow':
            add_AllCNN_block(4 * channel_reduction, input=True)
            if dropout:
                model.add(Dropout(0.2))
            add_AllCNN_block(4 * channel_reduction)
            add_AllCNN_block(16 * channel_reduction, strides=(2, 2))
            if dropout:
                model.add(Dropout(0.5))
            add_AllCNN_block(16 * channel_reduction)
            add_AllCNN_block(16 * channel_reduction)
            add_AllCNN_block(64 * channel_reduction, strides=(2, 2))
            if dropout:
                model.add(Dropout(0.5))
            add_AllCNN_block(64 * channel_reduction)
            add_AllCNN_block(64 * channel_reduction, filter_size=(1, 1))
            add_AllCNN_block(10 * channel_reduction, filter_size=(1, 1))
            model.add(Flatten())
            add_last_layers(no_layers=1)
    elif arg.network_type_coarse == 'FCN':
        if model_type_fine == '2_Layers':
            model.add(Flatten(input_shape=input_shape))
            add_FCN_block(3072 * channel_reduction)
            add_last_layers(no_layers=1)
        elif model_type_fine == '4_Layers':
            model.add(Flatten(input_shape=input_shape))
            add_FCN_block(3072 * channel_reduction)
            add_FCN_block(400 * channel_reduction)
            add_FCN_block(120 * channel_reduction)
            add_last_layers(no_layers=1)
        elif model_type_fine == '8_Layers':
            model.add(Flatten(input_shape=input_shape))
            add_FCN_block(4096 * channel_reduction)
            add_FCN_block(3072 * channel_reduction)
            add_FCN_block(2048 * channel_reduction)
            add_FCN_block(1024 * channel_reduction)
            add_FCN_block(512 * channel_reduction)
            add_FCN_block(256 * channel_reduction)
            add_FCN_block(128 * channel_reduction)
            add_last_layers(no_layers=1)
        elif model_type_fine == 'Wide':
            model.add(Flatten(input_shape=input_shape))
            # Section 1
            add_FCN_block(4096 * channel_reduction)
            if dropout:
                model.add(Dropout(0.2))
            add_FCN_block(4096 * channel_reduction)
            add_FCN_block(4096 * channel_reduction)
            if dropout:
                model.add(Dropout(0.5))
            # Section 2
            add_FCN_block(4096 * channel_reduction)
            add_FCN_block(4096 * channel_reduction)
            add_FCN_block(4096 * channel_reduction)
            if dropout:
                model.add(Dropout(0.5))
            # Section 3
            add_FCN_block(4096 * channel_reduction)
            add_FCN_block(4096 * channel_reduction)
            # Section 4
            add_last_layers()
    elif arg.network_type_coarse == 'ResNet':
        n = 3
        if model_type_fine == 'v1':
            depth = n * 6 + 2
            model = resnet_v1(input_shape=input_shape, depth=depth)
        elif model_type_fine == 'v2':
            depth = n * 9 + 2
            model = resnet_v2(input_shape=input_shape, depth=depth)
    elif arg.network_type_coarse == 'VGG':
        if model_type_fine == 'Without_Maxpooling':
            model.add(Conv2D(8 * channel_reduction, (3, 3), kernel_initializer='he_uniform', padding='same',
                             input_shape=input_shape))
            model.add(Activation(activation))
            add_VGG_block(16 * channel_reduction, maxpooling=False)
            add_VGG_block(32 * channel_reduction, maxpooling=False)
            model.add(Conv2D(64 * channel_reduction, (3, 3), kernel_initializer='he_uniform', padding='same'))
            model.add(Activation(activation))
            add_last_layers()
        elif model_type_fine == 'Narrow':
            add_VGG_block(4 * channel_reduction, input=True)
            add_VGG_block(16 * channel_reduction)
            add_VGG_block(64 * channel_reduction)
            add_last_layers()
        elif model_type_fine == 'Narrow_with_strides':
            add_VGG_block(4 * channel_reduction, maxpooling=False, input=True)
            add_VGG_block(16 * channel_reduction, strides=2)
            add_VGG_block(64 * channel_reduction, strides=2)
            add_last_layers()
        elif model_type_fine == 'Batchnorm':
            add_VGG_block(32 * channel_reduction, input=True, batchnorm=True)
            add_VGG_block(64 * channel_reduction, batchnorm=True)
            add_VGG_block(128 * channel_reduction, batchnorm=True)
            add_last_layers()
        elif model_type_fine == 'Standard':
            add_VGG_block(32 * channel_reduction, input=True)
            add_VGG_block(64 * channel_reduction)
            add_VGG_block(128 * channel_reduction)
            add_last_layers()
        elif model_type_fine == 'Standard_Av':
            add_VGG_block(32 * channel_reduction, input=True, maxpooling=False, avpooling=True)
            add_VGG_block(64 * channel_reduction, maxpooling=False, avpooling=True)
            add_VGG_block(128 * channel_reduction, maxpooling=False, avpooling=True)
            add_last_layers()
        elif model_type_fine == 'Deep':
            add_VGG_block(32 * channel_reduction, filter_size=(7, 7), input=True, maxpooling=False)
            add_VGG_block(64 * channel_reduction, filter_size=(5, 5), maxpooling=True)
            add_VGG_block(128 * channel_reduction, filter_size=(3, 3), maxpooling=False)
            add_VGG_block(256 * channel_reduction, filter_size=(3, 3), maxpooling=True)
            add_VGG_block(512 * channel_reduction, filter_size=(3, 3), maxpooling=False)
            add_VGG_block(1024 * channel_reduction, filter_size=(3, 3), maxpooling=True)
            add_last_layers(no_layers=3)
    elif arg.network_type_coarse == 'MNIST':
        if arg.network_type_fine == 'FCN4':
            model.add(Flatten(input_shape=input_shape))
            model.add(Dense(784))
            model.add(Activation(activation))
            model.add(Dense(200))
            model.add(Activation(activation))
            model.add(Dense(80))
            model.add(Activation(activation))
            model.add(Dense(10))
            model.add(Activation(Softmax()))
        elif arg.network_type_fine == 'FCN6':
            model.add(Flatten(input_shape=input_shape))
            model.add(Dense(2500))
            model.add(Activation(activation))
            model.add(Dense(2000))
            model.add(Activation(activation))
            model.add(Dense(1500))
            model.add(Activation(activation))
            model.add(Dense(1000))
            model.add(Activation(activation))
            model.add(Dense(500))
            model.add(Activation(activation))
            model.add(Dense(10))
            model.add(Activation(Softmax()))
        elif arg.network_type_fine == 'Convolutional':
            model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same',
                             kernel_regularizer=l2(0.001), input_shape=input_shape))
            model.add(Activation(activation))
            model.add(MaxPooling2D((2,2)))
            model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001)))
            model.add(Activation(activation))
            model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001)))
            model.add(Activation(activation))
            model.add(MaxPooling2D((2,2)))
            model.add(Flatten())
            model.add(Dense(64, kernel_regularizer=l2(0.001)))
            model.add(Activation(activation))
            model.add(Dense(10, kernel_regularizer=l2(0.001)))
            model.add(Activation(Softmax()))
    return model
