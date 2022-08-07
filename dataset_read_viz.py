

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import carla
import math


# Fetch the h5 files
dataset_path = '/home/vaydingul20/Documents/Codes/new_dataset_carla_9_10_1/expert/'
h5_files = [f for f in os.listdir(dataset_path) if f.endswith('.h5')]
while True:

    file = os.path.join(dataset_path, np.random.choice(h5_files))
    file_name = file.split('/')[-1]
    file = h5py.File(file, 'r')

    fig, ax = plt.subplots(1, 1)

    x = np.arange(0, 2*np.pi, 0.01)
    ax_image = ax.imshow(np.zeros_like(file['step_0/obs/central_rgb/data']))
    #ax_plot, = ax.plot([],[], 'o', color='red', markersize=5, alpha=0.5)
    #ax.set_xlim([0, 1000])
    #ax.set_ylim([0, 1000])
    ax.set_title("Test")

    def animate(i):
        ax_image.set_data(file[f'step_{int(i)}/obs/central_rgb/data'])

        command = file[f"step_{int(i)}/obs/gnss/command/"][()][0]
        throttle = file[f"step_{int(i)}/supervision/action/"][()][0]
        steer = file[f"step_{int(i)}/supervision/action/"][()][1]
        brake = file[f"step_{int(i)}/supervision/action/"][()][2]
        speed = file[f"step_{int(i)}/obs/speed/speed/"][()][0]
        ax.set_title(
            f'Step {i} - Command = {command} \n Throttle =  {throttle} - Steer =  {steer} - Brake =  {brake} \n Speed = {speed}')

        return ax_image, ax,

    ani = animation.FuncAnimation(
        fig, animate, np.arange(1, 2999), interval=1, blit=False)
    plt.show()
