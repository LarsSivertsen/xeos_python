import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

test = False
runs = [3000]
if(test):
    run_folder = "runs/test_runs/"
else:
    run_folder = "runs/"
i = 0
imax = 30000
for run_number in runs:
    filenames = []
    with open(run_folder+"run_"+str(run_number)+"/filenames.txt","r") as filenames:
        filenames.readline()

        for filename in filenames:
            with open(run_folder+"run_"+str(run_number)+"/TOV/"+filename[:10],"r") as MR:
                MR.readline()
                M_vec = []
                R_vec = []
                Lambda_vec = []
                P_c_vec = []
                rho_c_vec = []
                for line in MR:
                    M,R,Lambda,P_c,rho_c=line.split()
                    M = float(M)
                    R = float(R)
                    Lambda = float(Lambda)
                    P_c = float(P_c)
                    rho_c = float(rho_c)
                    M_vec.append(M)
                    R_vec.append(R)
                    Lambda_vec.append(Lambda)
                    P_c_vec.append(P_c)
                    rho_c_vec.append(rho_c)

                M_vec = np.array(M_vec)
                R_vec = np.array(R_vec)
                Lambda_vec = np.array(Lambda_vec)
                P_c_vec = np.array(P_c_vec)
                rho_c_vec = np.array(rho_c_vec)
                plt.figure("MR all",figsize=(10,10))
                plt.plot(R_vec,M_vec,color="grey",alpha=0.1)
                plt.figure("Lambda M all",figsize=(10,10))
                plt.plot(Lambda_vec,M_vec,color="grey",alpha=0.1)
                plt.figure("P_c M all",figsize=(10,10))
                plt.plot(P_c_vec,M_vec,color="grey",alpha=0.1)
                plt.figure("M of rho_c all")
                plt.plot(rho_c_vec,M_vec,color="grey",alpha=0.1)
            with open(run_folder+"run_"+str(run_number)+"/EoS/"+filename[:10],"r") as EoS:
                EoS.readline()
                e_vec = []
                P_vec = []
                rho_vec = []
                v2_vec = []
                mu_q_vec = []
                mu_e_vec = []
                for line in EoS:
                    e,P,rho,v2,mu_q,mu_e=line.split()
                    e = float(e)
                    P = float(P)
                    rho = float(rho)
                    v2 = float(v2)
                    mu_q = float(mu_q)
                    mu_e = float(mu_e)
                    e_vec.append(e)
                    P_vec.append(P)
                    rho_vec.append(rho)
                    v2_vec.append(v2)
                    mu_q_vec.append(mu_q)
                    mu_e_vec.append(mu_e)
                e_vec = np.array(e_vec)
                P_vec = np.array(P_vec)
                rho_vec = np.array(rho_vec)
                v2_vec = np.array(v2_vec)
                mu_q_vec = np.array(mu_q_vec)
                mu_e_vec = np.array(mu_e_vec)
                plt.figure("P of e all",figsize=(10,10))
                plt.plot(e_vec,P_vec,color="grey",alpha=0.1)
                plt.figure("v2 of rho all",figsize=(10,10))
                plt.plot(rho_vec,v2_vec,color="grey",alpha=0.1)
                plt.figure("P of rho all",figsize=(10,10))
                plt.plot(rho_vec,P_vec,color="grey",alpha=0.1)
                #plt.figure("v2 of rho all",figsize=(10,10))
                #plt.plot(rho_vec,v2_vec,color="grey",alpha=0.1)
            i+=1
            if(i>imax):
                break

plt.figure("MR all")
plt.xlabel("R[km]")
plt.ylabel("M[M$_\\odot$]")
if(len(runs)==1):
    run_number = runs[0]
    plt.title("MR-relations, run "+str(run_number))
    plt.savefig(run_folder+"run_"+str(run_number)+"/MR_all.pdf")
else:
    plt.title("MR-relations, runs "+str(runs[0])+"-"+str(runs[-1]))
    plt.savefig(run_folder+"MR_all.pdf")


plt.figure("Lambda M all")
plt.xlabel("$\\Lambda$")
plt.ylabel("M[M$_\\odot$]")
plt.xscale("log")
if(len(runs)==1):
    run_number = runs[0]
    plt.title("M$\\Lambda$-relations, run "+str(run_number))
    plt.savefig(run_folder+"run_"+str(run_number)+"/Lambda_M_all.pdf")
else:
    plt.title("M$P_c$-relations, runs "+str(runs[0])+"-"+str(runs[-1]))
    plt.savefig(run_folder+"Lambda_M_all.pdf")


plt.figure("P_c M all")
plt.xlabel("$P_c$[MeV/fm$^3$]")
plt.ylabel("M[M$_\\odot$]")
plt.xscale("log")
if(len(runs)==1):
    run_number = runs[0]
    plt.title("M$P_c$-relations, run "+str(run_number))
    plt.savefig(run_folder+"run_"+str(run_number)+"/P_c_M_all.pdf")
else:
    plt.title("M$P_c$-relations, runs "+str(runs[0])+"-"+str(runs[-1]))
    plt.savefig(run_folder+"P_c_M_all.pdf")


plt.figure("P of e all")
plt.xlim(0,6000)
plt.ylim(0,1400)
plt.xlabel("$e$[MeV/fm$^3$]")
plt.ylabel("$P$[MeV/fm$^{3}$]")
if(len(runs)==1):
    run_number = runs[0]
    plt.title("Pe-relations, run "+str(run_number))
    plt.savefig(run_folder+"run_"+str(run_number)+"/P_of_e_all.pdf")
else:
    plt.title("Pe-relations, runs "+str(runs[0])+"-"+str(runs[-1]))
    plt.savefig(run_folder+"P_of_e_all.pdf")

plt.figure("P of rho all")
plt.xlim(0,3)
plt.ylim(0,1400)
plt.xlabel("$\\rho$[fm$^3$]")
plt.ylabel("$P$[MeV/fm$^{3}$]")
if(len(runs)==1):
    run_number = runs[0]
    plt.title("Prho-relations, run "+str(run_number))
    plt.savefig(run_folder+"run_"+str(run_number)+"/P_of_rho_all.pdf")
else:
    plt.title("Prho-relations, runs "+str(runs[0])+"-"+str(runs[-1]))
    plt.savefig(run_folder+"P_of_rho_all.pdf")


plt.figure("M of rho_c all")
plt.xlim(0,5)
plt.ylim(0,2.2)
plt.xlabel("$\\rho_c$[MeV/fm$^3$]")
plt.ylabel("M[$M_\\odot$]")
if(len(runs)==1):
    run_number = runs[0]
    plt.title("Mrhoc-relations, run "+str(run_number))
    plt.savefig(run_folder+"run_"+str(run_number)+"/M_of_rho_c_all.pdf")
else:
    plt.title("Mrhoc-relations, runs "+str(runs[0])+"-"+str(runs[-1]))
    plt.savefig(run_folder+"M_of_rho_c_all.pdf")


plt.show()
