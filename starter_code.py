import numpy as np 
class HMM:

    def __init__(self, N, T_prob, B_list):
        '''
        Args:
            N: The grid side length
            T_porb: list of [p_right, p_left, p_up, p_down, p_same]
            B_list: contains 4 Bs correponding to 4 sensors
                each B is 1d np array of shape: B: [N^2]
                the sensor matrices are in order of S1, S2, S3, S4
        '''
        assert N > 0
        assert len(T_prob) == 5
        assert len(B_list) == 4
        for B in B_list:
            assert B.shape[0] == N*N 

         
        #side lenght
        self.N = N 

        #number of states
        self.n = (self.N**2)

        #[n x n]
        self.T = self.create_T(self.T_prob) 
        
        #the sensor index -> sensor symbol
        self._sensor_symbol = self.create_sensor_symbol()
        
        #the sensor grid
        self.sensor_grid = [[1,9,1,9], [7,15,1,9], [7,15,7,15], [1,9,7,15]]

        #[n x 16]
        self.B_list = B_list
        self.B = self.create_B(self.B_list)

        #[n]
        self.rho = np.zeros(self.n)
        self.rho[0] = 1 
    
    def create_sensor_symbol(self):

        sensor_symbol = []
        for s in range(16):
            symbol = []
            for i in range(4):
                symbol.append(s // int(2**(4-i-1)))
                s = s % (int(2**(4-i-1)))
            sensor_symbol.append(symbol)
        return np.array(sensor_symbol)

    #create the matrix
    def create_B(self, B_list):
        '''
        Args:
            B_list: list of 4 sensor Bs of shape [self.n]
        Return:
            B: [n,16]
        '''
        B = np.ones((self.n, 16))
        for j in range(16):
            s = self.sensor_symbol[j]
            for i, k in enumerate(s):
                B[:,j] *= (k*B_list[i] + (1-k)*(1-B_list[i]))
        return B

    def create_T(self, T_porb):
        
        #get the probabilities
        pr, pl, pu, pd, ps = T_porb[0], T_porb[1], T_porb[2], T_porb[3], T_porb[4]
        T = np.zeros((self.n, self.n))
        for i in range(self.n):
            
            x = (i % self.N) + 1 
            y = (i // self.N) + 1
            
            #self transition
            T[i,i] = ps

            #move to right
            if(x + 1 <= N):
                T[i,i+1] = pr
            else:
                T[i,i] += pr
            
            #move left
            if(x - 1 > 0):
                T[i,i-1] = pl
            else:
                T[i,i] += pl
            
            #move up
            if(y + 1 <= N):
                T[i,i+N] = pu
            else:
                T[i,i] += pu

            #move down
            if(y - 1 > 0):
                T[i,i-N] = pd
            else:
                T[i,i] += pd
        return T



    def sample(self, t):
        '''
        Args:
            t: integet, trajectory length
        '''
        pass
        
    def forward_inference(self, obs):
        '''
        Args:
            obs: np.array of shape [T]
        '''
        
    def e_step(self, alpha, beta):
        '''
        Args:
            alpha: [R, T, self.n]
            beta: [R, T, self.n]
        '''
        pass
    
    def m_step(self, ksi, gamma):
        '''
        Args:
            T: Transition matrix [self.n , self.n]
            B: Emission matrix [self.n, 16]
            rho: Rho [self.n]
            obs: R observation sequnece [R, t]
        '''
        pass
                     
    def baum_welch(self, obs):
        '''
        Args:
            obs: [R, T]
        '''
        pass
    
    def viterbi(self, obs):
        '''
        Args:
            obs: [R, T]
        '''
        pass

