import random
import numpy as np
import matplotlib.pyplot as plt

def envelope(Delta):
    x = Delta
    a = 200
    return np.where(x<a,(2/5*x+100,x**2*6.4e-4-80*6.4e-4*x+315),(2/5*x+100,2/5*x+250.36))

def make_parameter_file(filename,N,parameter_ranges,distributions=["gauss","gauss","gauss","gauss"],stds=3):
    #Parameters:
    #B,Delta,m_s,c
    parameters = np.zeros((4,N))
    for j in range(N):
        if(distributions[1] == "gauss"):
            mean_Delta = (parameter_ranges[1][1]+parameter_ranges[1][0])/2
            sigma_Delta = (mean-parameter_ranges[1][0])/stds
            Delta = random.gauss(mu=mean_Delta,sigma=sigma_Delta)
        elif(distributions[1] == "uniform"):
            Delta = random.uniform(parameter_ranges[1][0],parameter_ranges[1][1])

        else:
            print("No support for distribution: "+distribution[3])
            return

        if(distributions[0] == "gauss"):
            mean_B = (envelope(Delta).mean())/2
            sigma_B = (mean-envelope(Delta)[0])/stds
            B = random.gauss(mu=mean_B,sigma=sigma_B)

        elif(distributions[0] == "uniform"):
            B = random.uniform(envelope(Delta)[0],envelope(Delta)[1])

        else:
            print("No support for distribution: "+distribution[3])
            return

        if(distributions[2] == "gauss"):
            mean_m_s = (parameter_ranges[2][1]+parameter_ranges[2][0])/2
            sigma_m_s = (mean-parameter_ranges[2][0])/stds
            m_s = random.gauss(mu=mean_m_s,sigma=sigma_m_s)
        elif(distributions[2] == "uniform"):
            m_s = random.uniform(parameter_ranges[2][0],parameter_ranges[2][1])

        else:
            print("No support for distribution: "+distribution[3])
            return

        if(distributions[3] == "gauss"):
            mean_c = (parameter_ranges[3][1]+parameter_ranges[3][0])/2
            sigma_c = (mean-parameter_ranges[3][0])/stds
            c = random.gauss(mu=mean_c,sigma=sigma_c)
        elif(distributions[3] == "uniform"):
            c =  random.uniform(parameter_ranges[3][0],parameter_ranges[3][1])

        else:
            print("No support for distribution: "+distribution[3])
            return

        parameters[:,j] = B,Delta,m_s,c

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




'''
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
'''

