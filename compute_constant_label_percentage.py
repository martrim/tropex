import pickle
from Utilities.Parser import parse_arguments
from Utilities.Saver import get_saving_directory
import os

arg = parse_arguments()
save_dir = get_saving_directory(arg)
func_dir = os.path.join(save_dir, 'Training')
file_name = 'interpolation.pkl'
with open(os.path.join(func_dir, file_name), 'rb') as f:
    statistics = pickle.load(f)

print(statistics['no_no_label_changes']/(statistics['no_no_label_changes']+len(statistics['distances_at_first_label_changes'])))