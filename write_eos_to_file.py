import numpy as np

P_conv = 8.96057e-7 #Convert between MeV/fm^3 to M_sol/km^3

def write_crust_to_file(status,filename,filename_crust):
    file = open(filename,'w')
    file.write("status: "+str(status)+"\n")
    file.write("e[MeV/fm^3], P[MeV/fm^3], rho[fm^-3] \n")
    file_crust = open(filename_crust,'r')

    lines = []
    l = 0
    for line in file_crust:
        lines.append(line)
        l+=1
    for i in range(l-1,0,-1):
        lin = lines[i].split()
        e = np.float64(lin[0])/P_conv
        P = np.float64(lin[1])/P_conv
        rho = lin[2]
        file.write(str(e)+" "+str(P)+" "+rho+"\n")
    file.close()
    file_crust.close()


def write_EoS_to_file(eos,filename,crust=True,filename_crust="nveos.in"):

    if(crust==True):
        write_crust_to_file(eos.status,filename,filename_crust)
        file = open(filename,'a')
    else:
        file = open(filename,'w')

    e_vec = eos.e_vec
    P_vec = eos.P_vec
    rho_vec = eos.rho_vec



    for i in range(len(e_vec)):
        P = P_vec[i]
        e = e_vec[i]
        rho = rho_vec[i]
        if(P!=0):
            file.write(str(e)+" "+str(P)+" "+str(rho)+"\n")

    file.close()

    return

def read_EoS_from_file(filename):
    file = open(filename,'r')
    e_vec = []
    P_vec = []
    rho_vec = []

    file.readline()
    file.readline()
    for line in file:
        line  = line.split()
        P = np.float64(line[1])
        e = np.float64(line[0])
        rho = np.float64(line[2])
        P_vec.append(P)
        e_vec.append(e)
        rho_vec.append(rho)
    file.close()
    return np.array(P_vec),np.array(e_vec),np.array(rho_vec)

def read_parameters_from_file(filename):
    file = open(filename,'r')
    B_vec = []
    Delta_vec = []
    m_s_vec = []
    file.readline()
    for line in file:
        line  = line.split()
        B = np.float64(line[0])
        Delta = np.float64(line[1])
        m_s = np.float64(line[2])
        B_vec.append(B)
        Delta_vec.append(Delta)
        m_s_vec.append(m_s)
    file.close()
    return np.array(B_vec),np.array(Delta_vec),np.array(m_s_vec)

def write_MR_to_file(eos,filename):
    M_vec = eos.M_vec
    R_vec = eos.R_vec
    Lambda_vec = eos.Lambda_vec
    P_c_vec = eos.P_c_vec

    file = open(filename,'w')

    file.write("M[M_sun], R[km], Lambda[1], P_c[MeV/fm^3] \n")

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
        M = np.float64(line[0])
        R = np.float64(line[1])
        Lambda = np.float64(line[2])
        P_c = np.float(line[3])

        M_vec.append(M)
        R_vec.append(R)
        Lambda_vec.append(Lambda)
        P_c_vec.append(P_c)
    file.close()
    return np.array(M_vec),np.array(R_vec),np.array(Lambda_vec),np.array(P_c_vec)


