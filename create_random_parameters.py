import random
import numpy as np
import matplotlib.pyplot as plt

def make_parameter_file(filename,N,parameter_ranges,distributions=["gauss","gauss","gauss","gauss"],stds=3):
    parameters = np.zeros((4,N))
    for j in range(N):
        for i in range(4):
            if(distributions[i] == "gauss"):
                mean = (parameter_ranges[i][1]+parameter_ranges[i][0])/2
                sigma = (mean-parameter_ranges[i][0])/stds
                parameters[i,j] = random.gauss(mu=mean,sigma=sigma)
            elif(distributions[i] == "uniform"):
                parameters[i,j] = random.uniform(parameter_ranges[i][0],parameter_ranges[i][1])
            else:
                print("distribution not supported")
                return

    file = open(filename,'w')
    file.write("B^{1/4}[MeV], Delta[MeV], m_s[MeV], c[1] \n")
    for i in range(N):
        file.write(str(parameters[0,i])+" "+str(parameters[1,i])+" "+str(parameters[2,i])+" "+str(parameters[3,i])+"\n")
    file.close()
    return

def read_parameters_from_file(filename):
    file = open(filename,'r')
    B_vec = []
    Delta_vec = []
    m_s_vec = []
    c_vec = []
    file.readline()
    for line in file:
        B,Delta,m_s,c = line.split()
        B_vec.append(np.float64(B))
        Delta_vec.append(np.float64(Delta))
        m_s_vec.append(np.float64(m_s))
        c_vec.append(np.float64(c))
    file.close()
    return np.array(B_vec),np.array(Delta_vec),np.array(m_s_vec),np.array(c_vec)





