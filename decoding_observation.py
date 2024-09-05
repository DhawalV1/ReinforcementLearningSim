import numpy as np
from helper import *
np.random.seed(21)

grid_size = 15
num_steps = 30
# Function to get the next position based on current position


# Simulate the trajectory
trajectory_sample = []
for i in range(20):
    current_position = [1,1]
    trajectory = [current_position]
    for _ in range(29):
        next_position = sample_next_position(current_position)
        trajectory.append(next_position)
        current_position = next_position 
    
    trajectory_sample.append(trajectory)

B_list = B_list()

B = create_B(B_list)

sensor_read_sample = sampling_obs(trajectory_sample)


num_states = 225
num_observations = 16
num_time_steps = 30  

A = create_T([0.4,0.1,0.3,0.1,0.1])


# Viterbi algorithm implementation
def viterbi_algorithm(A, B, observations):
    num_states = A.shape[0]
    num_time_steps = len(observations)
    # Initialize the Viterbi matrix and backpointer matrix
    viterbi_matrix = np.zeros((num_states, num_time_steps))
    backpointer_matrix = np.zeros((num_states, num_time_steps), dtype=int)
    pi_s = np.zeros(225)
    pi_s[0] = 1
    # Initialize the first column of the Viterbi matrix
    for s in range(225):
        viterbi_matrix[s,0] = pi_s[s]*B[s, observations[1]]
        backpointer_matrix[s,0] = 0

    # Viterbi algorithm dynamic programming
    for t in range(1, num_time_steps):
        for s in range(num_states):
            arr = []
            for s_ in range(num_states):
                arr.append(viterbi_matrix[s_,t-1]*A[s,s_]*B[s,observations[t]])
            viterbi_matrix[s,t] = max(arr)
            backpointer_matrix[s,t] = np.argmax(arr)

    # Backtrack to find the most probable path
    best_path = np.zeros(num_time_steps, dtype=int)
    best_path[-1] = np.argmax(viterbi_matrix[:, -1])
    for t in range(num_time_steps - 2, -1, -1):
        best_path[t] = backpointer_matrix[best_path[t + 1], t + 1]
    return best_path

# Run the Viterbi algorithm

observation_holder = []
for i in range(len(sensor_read_sample)):
    observations=[]
    for t in sensor_read_sample[i]:
        observations.append(sum([m*2**l for l,m in enumerate(t[::-1])]))
    observation_holder.append(observations)

# reporting the decoded path for each observation sequence

decoded_path = []

for i in range(20):

    most_probable_path = viterbi_algorithm(A, B, observation_holder[i])
    most_probable_path_readable = [[state // 15+1, state % 15+1] for state in most_probable_path]
    
    print("decoded path by observation{}".format(i+1),most_probable_path_readable)
    decoded_path.append(most_probable_path_readable)
    
# mean manhattan distances

mean_manhattan = []
for i in range(30):
    manhattan_dist = 0
    for j in range(20):
        a,b = trajectory_sample[j][i]
        x,y = decoded_path[j][i]

        manhattan_dist += abs(a-x) + abs(b-y)
    mean_manhattan.append(manhattan_dist/20)

from grid import Grid
for i in range(20):
    grid = Grid()
    grid.draw_path(trajectory_sample[i],color = [1,1,1])
    grid.draw_path(decoded_path[i],color = [0,0,0])

    grid.show('compare{}.png'.format(i))


import matplotlib.pyplot as plt
import os
time = [i for i in range(1,31)]  
plt.figure(figsize=(8, 6))
plt.plot(time, mean_manhattan, label='Linear Data')
plt.title('Mean_manhattan_dist vs Time')
plt.xlabel('Time-step')
plt.ylabel('mean dist')
plt.legend()
plt.grid(True)
plt.savefig('mean_manhattan_dist_plot.png')  # Save in the current directory


