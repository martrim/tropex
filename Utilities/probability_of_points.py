import numpy as np
import os
import time
from Utilities.Saver import create_directory


def load_or_initialise(path):
    if os.path.isfile(path):
        return list(np.load(path))
    else:
        return []

def convert_time_format(seconds):
    seconds = round(seconds)
    hours = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    return str(hours) + ' hours, ' + str(minutes) + ' minutes, ' + str(seconds) + ' seconds.'


start = time.time()
data_dimension = 16
no_data_points = 10**7
epsilon_power = 3
epsilon = 10**(-epsilon_power)

save_dir = create_directory(os.getcwd(), 'Probability_of_Points')
ending = '_epsilon_' + str(epsilon_power) + '.npy'
min_path = os.path.join(save_dir, 'min' + ending)
max_path = os.path.join(save_dir, 'max' + ending)
mean_path = os.path.join(save_dir, 'mean' + ending)
median_path = os.path.join(save_dir, 'median' + ending)
std_path = os.path.join(save_dir, 'std' + ending)

min = load_or_initialise(min_path)
max = load_or_initialise(max_path)
mean = load_or_initialise(mean_path)
median = load_or_initialise(median_path)
std = load_or_initialise(std_path)

previously_completed_runs = len(min)

for i in range(previously_completed_runs, 10 ** 10 // 10 ** 7):
    x = np.random.standard_normal(size=(no_data_points, data_dimension))
    x_norm_squared = np.linalg.norm(x, axis=1) ** 2
    log_constant = data_dimension * np.log10(np.sqrt(2/np.pi)*epsilon)
    log_probability = log_constant + (-1/2 * x_norm_squared) +  np.log10(1 + 1/6*(x_norm_squared - data_dimension*(epsilon**2)))
    min.append(np.min(log_probability))
    max.append(np.max(log_probability))
    mean.append(np.mean(log_probability))
    median.append(np.median(log_probability))
    std.append(np.std(log_probability))
    if i % 20 == 0:
        np.save(min_path, min)
        np.save(max_path, max)
        np.save(mean_path, mean)
        np.save(median_path, median)
        np.save(std_path, std)
    time_passed = time.time() - start
    completed_runs = i - previously_completed_runs + 1
    runs_to_go = 10 ** 10 // 10 ** 7 - i - 1
    print('Finished run ' + str(i) + ' of ' + str(10 ** 10 // 10 ** 7) + '.')
    time_per_completed_run = time_passed/completed_runs
    time_left = time_per_completed_run * runs_to_go
    print('Expected time left: ' + convert_time_format(time_left))

total_min = np.min(min)
print('total min: ' + str(total_min))
total_max = np.max(max)
print('total max: ' + str(total_max))
total_mean = np.mean(mean)
print('total mean: ' + str(total_mean))
total_median = np.median(median)
print('total median: ' + str(total_median))
total_std = np.mean(std)
print('total std: ' + str(total_std))

print('Time taken:')
print(time.time() - start)
a = 5