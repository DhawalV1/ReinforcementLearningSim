
import numpy as np
from helper import *

np.random.seed(21)

# create emission probablity matrix

B_list = B_list()

B = create_B(B_list)

grid_size = 15
num_steps = 30

# Simulate the trajectory
trajectory_sample = []
for i in range(20):
    # a = np.random.randint(0,225)
    current_position = [1,1]
    trajectory = [current_position]
    for _ in range(29):
        next_position = sample_next_position(current_position)
        trajectory.append(next_position)
        current_position = next_position 
    # print("Sampled state trajectory:", trajectory)
    trajectory_sample.append(trajectory)

# extract observations
sensor_read_sample = sampling_obs(trajectory_sample)

# creates transition probablity matrix
T = create_T([0.4,0.1,0.3,0.1,0.1])

def forward_algo(obs,B,T):
  
  #alpha
  alpha = np.zeros((225, 30))

  #initialisation
  for s in range (0,225):
    if s==0:
      alpha[s][0]=1*B[s][bin2dec(obs[0])]

  # forward algo
  for obs_num in range (1, len(obs)):
    for s in range(0,225):
      for s_ in range(0,225):
        alpha[s][obs_num] += (alpha[s_][obs_num-1]*T[s_][s]*B[s][bin2dec(obs[obs_num])])

  return alpha

for k,observation in enumerate(sensor_read_sample):
  alpha = forward_algo(observation,B,T)
  likelihood=0
  for i in range(225):
    likelihood+=alpha[i][29]
  
  print('\nLikelihood of observation sequence{} is:'.format(k+1),f"{likelihood:.6e}")
