import numpy as np

def B_list():
    s1 = np.array([(18-(i-1)-(j-1))/18 if 1<=i<=9 and 1<=j<=9 else 0 for i in range(1,16) for j in range(1,16)])
    s2 = np.array([(18-(i-1)+(j-15))/18 if 1<=i<=9 and 7<=j<=15 else 0 for i in range(1,16) for j in range(1,16)])
    s3 = np.array([(18+(i-15)+(j-15))/18 if 7<=i<=15 and 7<=j<=15 else 0 for i in range(1,16) for j in range(1,16)])
    s4 = np.array([(18+(i-15)-(j-1))/18 if 7<=i<=15 and 1<=j<=9 else 0 for i in range(1,16) for j in range(1,16)])
    
    return [s1,s2,s3,s4]

def sample_next_position(current_position):
    i, j = current_position
    # Define possible moves with corresponding probabilities
    moves = [(i + 1, j, 0.4),  # Move right
                (i, j + 1, 0.3),  # Move up
                (i, j - 1, 0.1),  # Move down
                (i - 1, j, 0.1),  # Move left
                (i, j, 0.1)]      # Stay in place
    # Adjust probabilities for boundary conditions
    def condense_array(arr):
        arr = [(ni, nj, p) if 1 <= ni <= 15 and 1 <= nj <= 15 else (i, j, p)
                for ni, nj, p in arr]
        return [[i, j, sum(p for x, y, p in arr if (x, y) == (i, j))] for i, j in set((i, j) for i, j, _ in arr)]
    # Extract the positions and their corresponding probabilities
    positions, probabilities = zip(*[(pos[0:2], pos[2]) for pos in condense_array(moves)])
    next_position = positions[np.random.choice(len(positions), p=probabilities)]
    return next_position


# sampling the observation
def sampling_obs(trajectory_sample):
    sensor_read_sample = []
    b_list = [i.reshape(15,15) for i in B_list()]
    for traj in trajectory_sample:
        sensor_read = []
        for i,j in traj:
            
            pr = [0]*4
            pr[0]=b_list[0][i-1][j-1]
            pr[1]=b_list[1][i-1][j-1]
            pr[2]=b_list[2][i-1][j-1]
            pr[3]=b_list[3][i-1][j-1]
            obs = []
            for k in range(4):
                
                o = np.random.choice(np.arange(0,2),p=[1-pr[k],pr[k]])
                obs.append(int(o))
            sensor_read.append(obs[:])
        sensor_read_sample.append(sensor_read)
    return sensor_read_sample

# creating emission probablity matrix, B

def create_sensor_symbol():

    sensor_symbol = []
    for s in range(16):
        symbol = []
        for i in range(4):
            symbol.append(s // int(2**(4-i-1)))
            s = s % (int(2**(4-i-1)))
        sensor_symbol.append(symbol)
    return np.array(sensor_symbol)
    

def create_B(B_list):
    
    sensor_symbol = create_sensor_symbol()
    B = np.ones((225,16))
    for j in range(16):
        s = sensor_symbol[j]
        for i, k in enumerate(s):
            B[:,j] *= (k*B_list[i] + (1-k)*(1-B_list[i]))
    return B

def create_T(T_porb):
        
    #get the probabilities
    pr, pl, pu, pd, ps = T_porb[0], T_porb[1], T_porb[2], T_porb[3], T_porb[4]
    T = np.zeros((225, 225))
    for i in range(225):
        
        x = (i % 15) + 1 
        y = (i // 15) + 1
        
        #self transition
        T[i,i] = ps

        #move to right
        if(x + 1 <= 15):
            T[i,i+1] = pr
        else:
            T[i,i] += pr
        
        #move left
        if(x - 1 > 0):
            T[i,i-1] = pl
        else:
            T[i,i] += pl
        
        #move up
        if(y + 1 <= 15):
            T[i,i+15] = pu
        else:
            T[i,i] += pu

        #move down
        if(y - 1 > 0):
            T[i,i-15] = pd
        else:
            T[i,i] += pd
    return T

def bin2dec(obs_arr):
  return (obs_arr[3]+(obs_arr[2]*2)+(obs_arr[1]*4)+(obs_arr[0]*8))

def get_pos(state):
  return (((state[0]-1)*15)+state[1]-1)


def neighbour():
  neighbours = {}
  for i in range(225):
    neighbours[i] = [i]
    x = i//15 + 1
    y = i%15 + 1
    if(x-1>0):
      neighbours[i].append(get_pos([x-1,y]))
    if(x+1<16):
      neighbours[i].append(get_pos([x+1,y]))
    if(y-1>0):
      neighbours[i].append(get_pos([x,y-1]))
    if(y+1<16):
      neighbours[i].append(get_pos([x,y+1]))
  return neighbours

def kl_score(A,A_):
  import math
  kl=0
  for i in range(len(A)):
    for j in range(len(A[0])):
      
      epsilon = 1e-9
      A[i][j]=A[i][j]+epsilon
      A_[i][j] = A_[i][j]+epsilon
      
      kl = kl + (A[i][j] * (math.log(A[i][j] / A_[i][j])))
  avg_kl = kl/(len(A)*len(A[0]))
  return avg_kl
