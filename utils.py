import numpy as np
import os


def compute_mean_std(directory, sample_size):
    file_list = [file for file in os.listdir(directory) if file.endswith('.npy')]
    sample_files = np.random.choice(file_list, size=sample_size, replace=False)
    sample_data = []

    for file_name in sample_files:
        file_path = os.path.join(directory, file_name)
        data = np.load(file_path)
        sample_data.append(data)

    sample_data = np.concatenate(sample_data)
    mean = np.mean(sample_data)
    std = np.std(sample_data)

    return mean, std
