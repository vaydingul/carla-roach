import h5py
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt

CALCULATED_BEFORE = True

def calculate_command_distribution(h5_file, min_step=0, max_step=2999):
    f = h5py.File(h5_file, 'r')

    command_distribution = np.zeros((6,))

    for i in range(min_step, max_step):

        try:
        
            command = f[f'step_{int(i)}/obs/gnss/command'][()][0]
            command_distribution[int(command) - 1] += 1

        except KeyError:
            continue

    return command_distribution

# def calculate_weighted_sampling_coefficient(current_distribution, target_distribution):
    
#     current_distribution = np.array(current_distribution)
#     #current_distribution = current_distribution / np.sum(current_distribution)
#     target_distribution = np.array(target_distribution)
#     #target_distribution = target_distribution / np.sum(target_distribution)

#     n = current_distribution.shape[0]
#     assert n == target_distribution.shape[0], "Distributions must have the same length"

#     A = np.zeros((n, n))
#     B = np.zeros((n,))

#     #A = np.dot(current_distribution.reshape(-1, 1), target_distribution.reshape(-1, 1).T).T
#     #A -= np.diag(current_distribution)
#     for k in range(n):
#         for m in range(n):
#             if k == m:
#                 A[k, m] = current_distribution[k] * (target_distribution[m] - 1)
#             else:
#                 A[k, m] = current_distribution[k] * target_distribution[m]

#     coefficients = np.linalg.solve(A, B)

#     return coefficients

def calculate_weighted_sampling_coefficient(current_distribution, target_distribution):
    
    current_distribution = np.array(current_distribution)
    #current_distribution = current_distribution / np.sum(current_distribution)
    target_distribution = np.array(target_distribution)
    #target_distribution = target_distribution / np.sum(target_distribution)

    n = current_distribution.shape[0]
    assert n == target_distribution.shape[0], "Distributions must have the same length"

    A = np.zeros((n+1, n+1))
    B = np.zeros((n+1,))

    #A = np.dot(current_distribution.reshape(-1, 1), target_distribution.reshape(-1, 1).T).T
    #A -= np.diag(current_distribution)
    for k in range(n):
        A[k, k] = current_distribution[k]
        A[k, -1] = -target_distribution[k]
    A[n, :-1] = current_distribution
    A[n, -1] = -1

    coefficients = np.linalg.solve(A, B)

    return coefficients


def calculate_target_distribution(current_distribution, sampling_weights):

    current_distribution = np.array(current_distribution)
    current_distribution = current_distribution / np.sum(current_distribution)
    sampling_weights = np.array(sampling_weights)
    #sampling_weights = sampling_weights / np.sum(sampling_weights)
    target_distribution = current_distribution * sampling_weights
    target_distribution = target_distribution / np.sum(target_distribution)
    return target_distribution

if __name__ == "__main__":

    # Fetch the h5 files
    dataset_path = '/home/vaydingul/Documents/Codes/carla-dataset-detailed/expert/'
    if not CALCULATED_BEFORE:

        command_distribution_all = np.zeros((6,))

        
        for file in os.listdir(dataset_path):

            if file.endswith(".h5"):

                dataset_file = pathlib.Path(dataset_path) / file
                print(dataset_file)
                command_distribution = calculate_command_distribution(dataset_file)
                print(command_distribution)
                command_distribution_all += command_distribution
        np.savez("command_distribution.npz", command_distribution = command_distribution_all)


    else:

        command_distribution_all = np.load("command_distribution.npz")
        command_distribution_all = command_distribution_all["command_distribution"]

    commands = ["TURN LEFT",
    "TURN RIGHT",
    "STRAIGHT",
    "LANE FOLLOWING",
    "LEFT LANE CHANGE",
    "RIGHT LANE CHANGE"
    ]
    print(f"Current Distribution {command_distribution_all / np.sum(command_distribution_all)}")
    plt.figure()
    plt.bar(commands, command_distribution_all / np.sum(command_distribution_all))
    plt.xticks(commands, commands, rotation="vertical")
    plt.ylabel("Ratio of Number of Occurences")
    plt.title("Command Distribution")
    plt.tight_layout()
    plt.savefig(f"command_distribution_new.png")
    
    #coefficients = calculate_weighted_sampling_coefficient(current_distribution = command_distribution_all, target_distribution = command_distribution_all)
    #coefficients = calculate_weighted_sampling_coefficient([0.2, 0.8], [0.5, 0.5])
    #print(f"Coefficients: {coefficients}")

    weights = 1/command_distribution_all
    weights[3] *= 5
    weights *= 10000
    print(f"Weights: {weights}")
    target_distribution = calculate_target_distribution(command_distribution_all, weights)
    print(f"Target Distribution: {target_distribution}")
