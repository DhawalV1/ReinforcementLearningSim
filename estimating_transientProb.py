import numpy as np
from helper import *
import math

# part a)

B = create_B(B_list())
T = create_T([0.4,0.1,0.3,0.1,0.1]) 

def forward_inference (obs, T, B):
  #alpha
  alpha = np.zeros((225, len(obs)))

  # initialisation
  alpha[0][0]=1*B[0][bin2dec(obs[0])]

  neighbours = neighbour()
  for obs_num in range (1, len(obs)):
    o = bin2dec(obs[obs_num])
    for s in range(225):
 
      n = neighbours[s]
      alpha[s,obs_num] = np.sum(alpha[n,obs_num-1] * T[n, s]) * B[s, o]
  return alpha

def backward_inference(obs, T, B):
  
  beta = np.zeros((225, len(obs))) 
  
  # initialisation

  beta[:, -1] = 1

  neighbours = neighbour()
  for obs_num in range (len(obs)-2, -1, -1):
    o = bin2dec(obs[obs_num+1])           
    for s in range(0, 225):
      
      # Get the neighbors for state s
      n = neighbours[s]

      # calculating transition and observation probabilities
      
      trans_prob = T[s, n] 
      B_observ_prob = B[n, o]

      # Updating beta 

      beta[s, obs_num] = np.sum(B_observ_prob * trans_prob * beta[n, obs_num + 1])
  return beta

def e_step(alpha, beta, obs, T, B):

  num_states = 225
  num_obs = len(obs)

  # creating gamma 

  gamma = alpha * beta
  gamma /= np.sum(gamma, axis=0)
  obs_index = np.array([bin2dec(o) for o in obs[1:]])

  # creating ksi 

  ksi = np.zeros((num_obs-1, num_states, num_states))

  # computation of ksi

  for t in range(num_obs-1):
    alpha_ = alpha[:, t].reshape(-1, 1)  # Shape (num_states, 1)
    beta_ = beta[:, t+1].reshape(1, -1)  # Shape (1, num_states)
    B_ = B[:, obs_index[t]].reshape(1, -1)  # Shape (1, num_states)
    numerator = alpha_ * T * B_ * beta_
    denominator = np.sum(numerator)
    ksi[t] = numerator / denominator

  return gamma, ksi

def get_p_denominator(ksi):
  denominator = 0
  time_steps = 19 

    # Precompute the state numbers for all [x, y] pairs

  state_matrix = np.array([[get_pos([x, y]) for x in range(1, 16)] for y in range(1, 16)])

    # Calculate the sum of ksi[t][s][s] for all time steps and states
  for t in range(time_steps):
    for x in range(15):
      for y in range(15):
        s = state_matrix[x,y]
        # Add ksi[t][s][s]
        denominator += ksi[t, s, s]
        # Add ksi[t][s][neighbor_state] for each valid neighbor
        if x > 0:  # Left neighbor
          denominator += ksi[t, s, state_matrix[x, y - 1]]
        if x < 14:  # Right neighbor
          denominator += ksi[t, s, state_matrix[x, y + 1]]
        if y > 0:  # Top neighbor
          denominator += ksi[t, s, state_matrix[x - 1, y]]
        if y < 14:  # Bottom neighbor
          denominator += ksi[t, s, state_matrix[x + 1, y]]

  return denominator

def baum_welch(T, B, observation_seq):
  
  # parameters are initialized T, initialized B, and observations

  scores = []
  iter = 1
  for iteration in range(20):
    print('\n  Iternation Number: ', iter)
    numerator = [0,0,0,0,0] #right, up, down, left, same
    
    p = [0,0,0,0,0]
    
    for observation in observation_seq:
  
      #E-Step
      alpha = forward_inference(observation, T, B)
      beta  = backward_inference(observation, T, B)
      gamma, ksi = e_step(alpha, beta, observation, T, B)

      # M-Step
      #p_right
      index = np.arange(209)
      numerator[0] = numerator[0] + np.sum(ksi[:, index, index + 15]) 
      #p_left
      index = np.arange(15, 225)
      numerator[3] = numerator[3] + np.sum(ksi[:, index, index - 15])
      #p_up
      index = np.arange(225)
      index = index[index % 15 != 14]
      numerator[1] = numerator[1] + np.sum(ksi[:, index, index + 1])
      #p_down
      index = np.arange(225)
      index = index[index % 15 != 0]
      numerator[2] = numerator[2] + np.sum(ksi[:, index, index - 1]) 
      #p_same
      index = np.arange(225)
      numerator[4] = numerator[4] + np.sum(ksi[:, index, index ]) 
    
    denominator = np.sum(numerator[:5]) + 1e-10
    p[:5] = np.divide(numerator[:5], denominator)
    p = [float(x) for x in p]
    print('[right, up, down, left, stay] =', p)

    T_ = create_T(p)

    #calculate Divergence between T_true & newT
    score = kl_score(T, T_)
    print('Score for iteration:', score)
    scores.append(score)

    T = T_
    iter = iter + 1 
   
  print('Scores are ',scores)
  return T_, scores

############################################
np.random.seed(21)
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

############################################

T1 = create_T([0.2, 0.2, 0.2, 0.2, 0.2])

#executing baum-welch
predictedT, Tscores, = baum_welch(T1, B, sensor_read_sample)
