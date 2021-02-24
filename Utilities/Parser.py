import argparse
import os


def parse_arguments(print_args=True):
    parser = argparse.ArgumentParser(description='Parsing Arguments.')

    # Mandatory Arguments
    parser.add_argument('--gpu', default='0', type=str, help='gpu number (default: 0)')
    parser.add_argument('--epochs', default='all', choices=['none', 'all', 'special'],
                        help='specify which epochs are used')
    parser.add_argument('--data_set', default='CIFAR10', choices=['MNIST', 'CIFAR10', 'Fashion MNIST'],
                        help='type of architecture used for training (default: CIFAR10)')
    parser.add_argument('--data_type', default='training', choices=['training', 'test', 'random_hypercube', 'random_100'],
                        help="use training or test data")
    parser.add_argument('--network_type_coarse', default='FCN', choices=['-', 'AllCNN', 'FCN', 'ResNet', 'VGG', 'MNIST'],
                        help='type of architecture used for training (default: VGG)')
    parser.add_argument('--network_type_fine', default='8_Layers',
                        choices=['Standard', 'Narrow', 'Narrow_with_strides', '8_Layers', 'v1', 'Convolutional', 'FCN4', 'FCN6', 'Wide'],
                        help='name of network structure used for training (default: Standard)')
    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true',
                        help='specify if early stopping is used')
    parser.add_argument('--lr_schedule', dest='lr_schedule', action='store_true',
                        help='specify if learning rate schedule is used')
    parser.add_argument('--weight_decay', dest='weight_decay', action='store_true',
                        help='specify if weight decay is used (only for debugging)')

    # String arguments
    parser.add_argument('--network_number', default='0', choices=['0', '1', '2', '3', '4'],
                        help='number of saved network; the program loops through all numbers if "all" is selected')
    parser.add_argument('--network_number_2', default='1', choices=['0', '1', '2', '3', '4'],
                        help='number of saved network; the program loops through all numbers if "all" is selected')
    parser.add_argument('--activation_function', default='relu', choices=['leaky_relu', 'relu', 'split_relu'],
                        help='activation functions before the last layer (default: relu)')

    # Boolean Arguments (default: False)
    parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true',
                        help='specify if data augmentation is used')
    parser.add_argument('--dropout', dest='dropout', action='store_true',
                        help='specify if dropout is used')
    parser.add_argument('--gradient_tape', dest='gradient_tape', action='store_true',
                        help='specify if gradient tape is used')
    parser.add_argument('--temperature_scaling', dest='temperature_scaling', action='store_true',
                        help='specify if the temperature scaling should be used after every epoch')
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',
                        help='specify if tensorboard is used')
    parser.add_argument('--train_eagerly', dest='train_eagerly', action='store_true',
                        help='specify if eager training is used (only for debugging)')

    # Float Arguments
    # No temperature scaling when overheating = 0.
    parser.add_argument('--overheating_factor', default=1.0, type=float,
                        help='usual temperature scaling if 1.0 (default: 1.0)')
    parser.add_argument('--maximum_temperature', default=1.0, type=float,
                        help='maximum above which the temperature gets reset (default: 100.0)')
    parser.add_argument('--reset_temperature', default=1.0, type=float,
                        help='level to which the temperature gets reset (default: 100.0)')
    parser.add_argument('--initial_temperature', default=1.0, type=float,
                        help='temperature at the beginning of training (default: 1.0)')

    # Integer Arguments
    parser.add_argument('--batch_size', default=64, type=int,
                        help='the number of data points each training batch contains')
    parser.add_argument('--no_bins', default=15, type=int,
                        help='the number of bins for computing the ECE')
    parser.add_argument('--no_epochs', default=200, type=int,
                        help='the number of training epochs')

    # Tropical Arguments
    parser.add_argument('--extract_all_dimensions', dest='extract_all_dimensions', action='store_true',
                        help="if True: extract the linear functions corresponding to all dimensions")

    # Transformation Arguments
    parser.add_argument('--pos_or_neg', default='pos_and_neg', choices=['pos_and_neg', 'pos', 'neg'],
                        help='compute the numerator or the denominator of the tropical rational function')
    parser.add_argument('--data_points_lower', default=0, type=int, help="lower index of data points (default: 0)")
    parser.add_argument('--data_points_upper', default=50000, type=int,
                        help="upper index of data points (default: 100)")
    parser.add_argument('--save_intermediate', dest='save_intermediate', action='store_true',
                        help="saving all intermediate tropical functions")

    # Experiment Argument
    parser.add_argument('--mode', default='exp11_compare_linear_functions',
                        choices=['transformation', 'evaluation', 'exp1_count', 'exp2_val', 'exp3_compare',
                                 'exp4_variation', 'exp5_compute_translation_invariance', 'exp6_add_shifted_functions',
                                 'exp7_implications', 'exp8_extract_weight_matrices',
                                 'exp9_compute_coefficient_statistics', 'exp10_slide_extracted_function_over_image',
                                 'exp11_compare_linear_functions', 'exp12_compare_activation_patterns',
                                 'exp14_interpolation',
                                 'compute_averages', 'save_to_mat', 'save_linear_coefficients_to_mat',
                                 'compute_network_accuracies'],
                        help="name of experiment that is run")

    # Statistical computation arguments
    parser.add_argument('--fast', default=True, type=bool,
                        help='if fast: computes only the opt-values for the dictionary')

    # Exp3 arguments
    parser.add_argument('--lower_label', default=4, type=int,
                        help='asdf')
    parser.add_argument('--idx', default=100, type=int,
                        help='jkl')

    # For ResNet, use weight_decay = 1e-4, learning rate scheduler and batch_size=32.
    if "PYCHARM_HOSTED" in os.environ:
        print('Running in Pycharm.')
        # arg = parser.parse_args(['--early_stopping'])
        arg = parser.parse_args(['--weight_decay', '--lr_schedule'])
    else:
        arg = parser.parse_args()

    if print_args:
        start_bold = '\033[1m'
        end_bold = '\033[0m'
        print(start_bold + 'General Arguments' + end_bold)
        print('GPU: ' + arg.gpu)
        print('Data Set: ' + arg.data_set)
        print('Network Type Coarse: ' + arg.network_type_coarse)
        print('Network Type Fine: ' + arg.network_type_fine)
        print('Network Number: ' + arg.network_number)
        print('Activation Function: ' + arg.activation_function)
        print('\n')
        print(start_bold + 'Boolean Arguments' + end_bold)
        print('Data Augmentation: ' + str(arg.data_augmentation))
        print('Dropout: ' + str(arg.dropout))
        print('Early Stopping: ' + str(arg.early_stopping))
        print('Gradient Tape: ' + str(arg.gradient_tape))
        print('Learning Rate Schedule: ' + str(arg.lr_schedule))
        print('Epochs used: ' + str(arg.epochs))
        print('Temperature Scaling: ' + str(arg.temperature_scaling))
        print('Tensorboard: ' + str(arg.tensorboard))
        print('Train Eagerly: ' + str(arg.train_eagerly))
        print('Weight Decay: ' + str(arg.weight_decay))
        print('\n')
        print(start_bold + 'Float Arguments' + end_bold)
        print('Overheating Factor: ' + str(arg.overheating_factor))
        print('Maximum Temperature: ' + str(arg.maximum_temperature))
        print('Reset Temperature: ' + str(arg.reset_temperature))
        print('Initial Temperature: ' + str(arg.initial_temperature))
        print('\n')
        print(start_bold + 'Integer Arguments' + end_bold)
        print('Batch Size: ' + str(arg.batch_size))
        print('Number of Bins: ' + str(arg.no_bins))
        print('Number of Training Epochs: ' + str(arg.no_epochs))
        print('\n')

    return arg
