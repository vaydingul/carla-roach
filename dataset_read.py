

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Fetch the h5 files
dataset_path = '/home/vaydingul20/Documents/Codes/new_dataset_carla_9_10_1/expert/'
h5_files = [f for f in os.listdir(dataset_path) if f.endswith('.h5')][0:3]
while True:

    f = os.path.join(dataset_path, np.random.choice(h5_files))
    file_name = f.split('/')[-1]
    f = h5py.File(f, 'r')

    fig, ax = plt.subplots(1,1)

    x = np.arange(0, 2*np.pi, 0.01)
    ax_image = ax.imshow(np.zeros_like(f['step_0/obs/central_rgb/data']))
    ax.set_title("Test")


    def animate(i):
        ax_image.set_data(f['step_%d/obs/central_rgb/data' % i])
        print(i, file_name)
        ax.set_title(
            f'Step {i} - Command = {f["step_%d/obs/gnss/command/" % i].value[0]} \n Throttle =  {f["step_%d/supervision/action/" % i].value[0]} - Steer =  {f["step_%d/supervision/action/" % i].value[1] % i} - Brake =  {f["step_%d/supervision/action/" % i].value[2]} \n Speed = {f["step_%d/obs/speed/speed" % i].value[0]}')
        
        return ax_image, ax, 


    ani = animation.FuncAnimation(
        fig, animate, np.arange(1, 100), interval=1, blit=False)
    plt.show()
