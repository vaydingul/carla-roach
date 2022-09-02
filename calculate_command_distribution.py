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
    print(command_distribution_all / np.sum(command_distribution_all))
    plt.figure()
    plt.bar(commands, command_distribution_all / np.sum(command_distribution_all))
    plt.xticks(commands, commands, rotation="vertical")
    plt.ylabel("Ratio of Number of Occurences")
    plt.title("Command Distribution")
    plt.tight_layout()
    plt.savefig(f"command_distribution_new.png")
    