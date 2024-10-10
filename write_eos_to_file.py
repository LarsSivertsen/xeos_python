import numpy as np

P_conv = 8.96057e-7 #Convert between MeV/fm^3 to M_sol/km^3

def write_EoS_to_file(eos,filename):

    file = open(filename,'w')
    file.write("e[MeV/fm^3], P[MeV/fm^3], rho[fm^-3], v2[1], mu_q[MeV], mu_e[MeV]\n")

    mu_q_vec = eos.mu_q_vec
    mu_e_vec = eos.mu_e_vec
    e_vec = eos.e_vec
    P_vec = eos.P_vec
    rho_vec = eos.rho_vec
    v2_vec = eos.v2_vec
    for i in range(len(e_vec)):
        P = P_vec[i]
        e = e_vec[i]
        rho = rho_vec[i]
        v2 = v2_vec[i]
        mu_e = mu_e_vec[i]
        mu_q = mu_q_vec[i]
        if(P!=0):
            file.write(str(e)+" "+str(P)+" "+str(rho)+" "+str(v2)+" "+str(mu_q)+" "+str(mu_e)+"\n")
    file.close()

    return

def read_EoS_from_file(filename):
    file = open(filename,'r')
    e_vec = []
    P_vec = []
    rho_vec = []
    v2_vec = []
    mu_q_vec = []
    mu_e_vec = []

    file.readline()
    file.readline()
    for line in file:
        line  = line.split()
        P = float(line[1])
        e = float(line[0])
        rho = float(line[2])
        v2 = float(line[3])
        mu_q = float(line[4])
        mu_e = float(line[5])
        P_vec.append(P)
        e_vec.append(e)
        rho_vec.append(rho)
        v2_vec.append(v2)
        mu_q_vec.append(mu_q)
        mu_e_vec.append(mu_e)
    file.close()
    return np.array(P_vec),np.array(e_vec),np.array(rho_vec),np.array(v2_vec),np.array(mu_q_vec),np.array(mu_e_vec)


def write_MR_to_file(eos,filename):
    M_vec = eos.M_vec
    R_vec = eos.R_vec
    Lambda_vec = eos.Lambda_vec
    P_c_vec = eos.P_c_vec

    file = open(filename,'w')

    file.write("M[M_sun], R[km], Lambda[1], P_c[MeV/fm^3]\n")

    for i in range(len(M_vec)):
        M = M_vec[i]
        R = R_vec[i]
        Lambda = Lambda_vec[i]
        P_c = P_c_vec[i]

        file.write(str(M)+" "+str(R)+" "+str(Lambda)+" "+str(P_c)+"\n")

    file.close()

    return

def read_MR_from_file(filename):
    file = open(filename,'r')
    M_vec = []
    R_vec = []
    Lambda_vec = []
    P_c_vec = []

    file.readline()
    for line in file:
        line  = line.split()
        M = float(line[0])
        R = float(line[1])
        Lambda = float(line[2])
        P_c = float(line[3])

        M_vec.append(M)
        R_vec.append(R)
        Lambda_vec.append(Lambda)
        P_c_vec.append(P_c)
    file.close()
    return np.array(M_vec),np.array(R_vec),np.array(Lambda_vec),np.array(P_c_vec)


