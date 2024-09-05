# part a

import numpy as np
import matplotlib.pyplot as plt
from helper import *

B_list = B_list()

for i in range(4):
    B_list[i] = B_list[i].reshape(15, 15)
    plt.imshow(B_list[i], cmap='hot', interpolation='nearest')
    save_path = f'matrix_heatmap_{i}.png' 
    plt.savefig(save_path)


#part b

np.random.seed(21)

grid_size = 15

num_steps = 30

# Simulate the trajectory
trajectory_sample = []
for i in range(20):
    current_position = [1,1]
    trajectory = [current_position]
    for _ in range(29):
        next_position = sample_next_position(current_position)
        trajectory.append(next_position)
        current_position = next_position 
    print('trajectory{}:'.format(i), trajectory)
    trajectory_sample.append(trajectory)

#part c
# drawing each simulated path
from grid import Grid

for i in range(20):
    grid = Grid()
    grid.draw_path(trajectory_sample[i],color = [1,1,1])

    grid.show('trajectory{}.png'.format(i))

#part d

# sampling the observation

sensor_read_sample = sampling_obs(trajectory_sample)
for i in range(len(sensor_read_sample)):
    print("observation{}:".format(i+1),sensor_read_sample[i])








