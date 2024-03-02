import random 
import numpy as np
import matplotlib.pyplot as plt

def make_parameter_file(filename,N,parameter_ranges,distributions=["gauss","gauss","gauss"],stds=3):
    parameters = np.zeros((3,N))
    for j in range(N):
        for i in range(3):
            if(distributions[i] == "gauss"):
                mean = (parameter_ranges[i][1]+parameter_ranges[i][0])/2
                sigma = (mean-parameter_ranges[i][0])/stds
                parameters[i,j] = random.gauss(mu=mean,sigma=sigma)
    
    file = open(filename,'w')
    file.write("B^{1/4}[MeV], Delta[MeV], m_s[MeV] \n")
    for i in range(N):
        file.write(str(parameters[0,i])+" "+str(parameters[1,i])+" "+str(parameters[2,i])+"\n")
    file.close()
    return 

def read_parameters_from_file(filename):
    file = open(filename,'r')
    B_vec = []
    Delta_vec = []
    m_s_vec = []
    file.readline()
    for line in file:
        B,Delta,m_s = line.split()
        B_vec.append(np.float64(B))
        Delta_vec.append(np.float64(Delta))
        m_s_vec.append(np.float64(m_s))
    file.close()
    return np.array(B_vec),np.array(Delta_vec),np.array(m_s_vec)


'''

filename = "runs/parameters_test.txt"
N = 3
parameter_ranges = [[30,60],[90,1000],[0,30]]
make_parameter_file(filename,N,parameter_ranges)
B_vec,Delta_vec,m_s_vec = read_parameters_from_file(filename)
print(B_vec)
print(Delta_vec)
print(m_s_vec)


filename = "parameter_test"
N = 100000
parameter_ranges = [[0,10],[50,100],[300,1000]]
parameters = make_parameter_file(filename,N,parameter_ranges)
plt.figure()
plt.hist(parameters[0,:],bins = 100)
plt.figure()
plt.hist(parameters[1,:],bins = 100)
plt.figure()
plt.hist(parameters[2,:],bins = 100)
plt.show()
'''
