import time
import logging
import os
import sys
from Utilities.Saver import create_directory, get_saving_directory


def print_file_name(starting_or_ending):
    file_name = os.path.basename(sys.argv[0])
    start_string_1 = starting_or_ending.upper() + ' WITH PROGRAM'
    length_1 = len(start_string_1)
    start_string_2 = file_name
    length_2 = len(start_string_2)
    max_length = max(length_1, length_2)
    add_space_1 = max_length - length_1
    add_space_2 = max_length - length_2
    no_asterisks = 3
    no_spaces = 1
    total_line_length = max_length + (no_asterisks + no_spaces) * 2
    print('*' * total_line_length)
    asterisk_chain = '*' * no_asterisks
    if add_space_1 % 2 == 0:
        additional_space = 0
    else:
        additional_space = 1
    space_chain_1 = ' ' * (no_spaces + add_space_1//2 + additional_space)
    if add_space_2 % 2 == 0:
        additional_space = 0
    else:
        additional_space = 1
    space_chain_2 = ' ' * (no_spaces + add_space_2//2 + additional_space)
    print(asterisk_chain + space_chain_1 + start_string_1 + space_chain_1 + asterisk_chain)
    print(asterisk_chain + space_chain_2 + start_string_2 + space_chain_2 + asterisk_chain)
    print('*' * total_line_length)


def print_start():
    print_file_name('starting')
    start_time = time.time()
    print('Current time: ' + time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime()))
    return start_time


def print_end(start_time):
    print_file_name('ending')
    print('Current time: ' + time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime()))
    end_time = time.time()
    print('Time taken:')
    print(str((end_time - start_time) / 60) + ' minutes')
    return


def get_logger(arg):
    save_dir = get_saving_directory(arg)
    transformation_path = create_directory(save_dir, arg.data_type.capitalize())
    file_name = sys.argv[0].split('/')[-1].split('.')[0]
    if file_name == 'TropEx' or file_name == 'TropEx2' or file_name == 'TropEx_WIP':
        logger = logging.getLogger('TropEx_log')
        logger_path = os.path.join(transformation_path, 'TropEx.log')
    elif file_name == 'Evaluation':
        logger = logging.getLogger(arg.data_type.capitalize() + '_log')
        logger_path = os.path.join(transformation_path, arg.data_type + '.log')
    elif arg.mode == 'exp4_variation':
        logger = logging.getLogger('exp4_variation_log')
        logger_path = os.path.join(transformation_path, 'exp4_variation' + '.log')
    elif arg.mode ==  'exp6_add_shifted_functions':
        logger = logging.getLogger('exp6_add_shifted_functions_log')
        logger_path = os.path.join(transformation_path, 'exp6_add_shifted_functions' + '.log')
    elif arg.mode == 'exp9_compute_coefficient_statistics':
        logger = logging.getLogger('exp9_compute_coefficient_statistics_log')
        logger_path = os.path.join(transformation_path, 'exp9_compute_coefficient_statistics' + '.log')
    elif arg.mode == 'exp10_slide_extracted_function_over_image':
        logger = logging.getLogger('exp10_slide_extracted_function_over_image_log')
        logger_path = os.path.join(transformation_path, 'exp10_slide_extracted_function_over_image' + '.log')
    elif arg.mode == 'exp11_compare_linear_functions':
        logger = logging.getLogger('exp11_compare_linear_functions_log')
        logger_path = os.path.join(transformation_path, 'exp11_compare_linear_functions' + '.log')
    elif arg.mode == 'exp12_compare_activation_patterns':
        logger = logging.getLogger('exp12_compare_activation_patterns_log')
        logger_path = os.path.join(transformation_path, 'exp12_compare_activation_patterns' + '.log')
    elif arg.mode == 'save_linear_coefficients_to_mat':
        logger = logging.getLogger('save_linear_coefficients_to_mat_log')
        logger_path = os.path.join(transformation_path, 'save_linear_coefficients_to_mat' + '.log')
    elif arg.mode == 'exp14_interpolation':
        logger = logging.getLogger('interpolation_log')
        logger_path = os.path.join(transformation_path, 'interpolation' + '.log')
    elif arg.mode == 'save_to_mat':
        logger = logging.getLogger('save_to_mat_log')
        logger_path = os.path.join(transformation_path, 'save_to_mat' + '.log')
    elif arg.mode == 'compute_network_accuracies':
        logger = logging.getLogger('compute_network_accuracies_log')
        logger_path = os.path.join(transformation_path, 'compute_network_accuracies' + '.log')
    else:
        logger = None
        logger_path = ''
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logger_path, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("================================================")
    logger.info("Data Type: " + arg.data_type.capitalize())
    logger.info("Network Type: " + arg.network_type_coarse)
    logger.info("Network Name: " + arg.network_type_fine)
    logger.info("Lower Index of Data Points: {}".format(arg.data_points_lower))
    logger.info("Upper Index of Data Points: {}".format(arg.data_points_upper))
    return logger
