import EoS_classes
import matplotlib.pyplot as plt
import time
import numpy as np

testing_apr03 = True               #Check apr eos
testing_CFL = True                 #Check apr eos and intersection with apr
testing_total_eos_B = True          #Check total eos for different values of bag constant B
testing_total_eos_Delta = True      #Check total eos for different values of pairing gap Delta
testing_total_eos_m_s = True        #Check total eos for different values strange mass m_s
#Testing
if(testing_apr03==True):

    time_0 = time.perf_counter()
 
    N_apr03 = 1000
    eos = EoS_classes.apr03_EoS(N_apr03) 
    print("Time spent:", time.perf_counter()-time_0)

    plt.figure()
    plt.xlim(0,1)
    plt.ylim(-50,450)
    plt.xlabel("$\\rho$ [fm$^{-3}$]")
    plt.ylabel("E/A [MeV]")

    plt.plot(eos.rho_PNM_SNM_apr03_vec,eos.E_per_A_PNM_LDP_apr03_vec,label = "PNM, LDP",color="Black")
    plt.plot(eos.rho_PNM_SNM_apr03_vec,eos.E_per_A_PNM_HDP_apr03_vec,label = "PNM, HDP",color = "Red")
    plt.plot(eos.rho_PNM_SNM_apr03_vec,eos.E_per_A_SNM_LDP_apr03_vec,'--',label = "SNM, LDP", color = "Black")
    plt.plot(eos.rho_PNM_SNM_apr03_vec,eos.E_per_A_SNM_HDP_apr03_vec,'--',label = "SNM, HDP", color = "Red")
    plt.legend()
    plt.savefig("figures/tests/figures_xeos_note/E_per_A_PNM_and_SNM.pdf")

    plt.figure()
    plt.xlim(0,1)
    plt.ylim(-50,450)
    plt.xlabel("$\\rho$ [fm$^{-3}$]")
    plt.ylabel("E/A [MeV]")
    plt.plot(eos.rho_apr03_vec,eos.E_per_A_LDP_apr03_vec,label = "LDP",color = "Black")
    plt.plot(eos.rho_apr03_vec,eos.E_per_A_HDP_apr03_vec,label = "HDP",color = "Red")
    plt.legend()
    plt.savefig("figures/tests/figures_xeos_note/E_per_A_beta_equillibrium.pdf")
    
    plt.figure()
    plt.xlim(0,1.2)
    plt.ylim(0,0.2)
    plt.xlabel("$\\rho$ [fm$^{-3}$]")
    plt.ylabel("$x_p$")
    plt.plot(eos.rho_apr03_vec,eos.x_p_LDP_apr03_vec,'*',label = "LDP", color = "Blue")
    plt.plot(eos.rho_apr03_vec,eos.x_p_HDP_apr03_vec,'*',label = "HDP", color = "Red")
    plt.plot(eos.rho_apr03_combined_vec,eos.x_p_apr03_combined_vec,label = "APR03", color="Black")
    plt.legend()
    plt.savefig("figures/tests/figures_xeos_note/proton_fraction.pdf")
    
    plt.figure()
    plt.xlim(0,1)
    plt.ylim(0,600)
    plt.xlabel("$\\rho_0$[fm$^{-3}$]")
    plt.ylabel("Energy density [MeV/fm$^3$]")
    
    plt.plot(eos.rho_apr03_vec,eos.e_LDP_apr03_vec,'*',color = "Blue",label = "LDP")
    plt.plot(eos.rho_apr03_vec,eos.e_HDP_apr03_vec,'*',color = "Red",label = "HDP")
    plt.plot(eos.rho_apr03_combined_vec,eos.e_apr03_combined_vec,color = "Black",label = "APR03")
    plt.legend()
    plt.savefig("figures/tests/figures_xeos_note/e_of_rho.pdf")

    plt.figure()
    plt.xlim(0,100)
    plt.ylim(0,600)
    plt.xlabel("Pressure [MeV/fm$^3$]")
    plt.ylabel("Energy density [MeV/fm$^3$]")
    plt.plot(eos.P_LDP_apr03_vec,eos.e_LDP_apr03_vec,'*',label = "LDP",color = "Blue")
    plt.plot(eos.P_HDP_apr03_vec,eos.e_HDP_apr03_vec,'*',label = "HDP",color = "Red")
    plt.plot(eos.P_apr03_combined_vec,eos.e_apr03_combined_vec,label = "APR03", color = "Black")
    plt.legend()
    plt.savefig("figures/tests/figures_xeos_note/e_of_P.pdf")

    plt.figure()
    plt.xlim(0,1)
    plt.ylim(800,2100)
    plt.xlabel("$\\rho$[fm$^{-3}$]")
    plt.ylabel("$\\mu$[MeV]")
    plt.plot(eos.rho_apr03_vec,eos.mu_n_LDP_apr03_vec,'*',label = "$\\mu_n$, LDP")
    plt.plot(eos.rho_apr03_vec,eos.mu_n_HDP_apr03_vec,'*',label = "$\\mu_n$, HDP")
    plt.plot(eos.rho_apr03_vec,eos.mu_p_LDP_apr03_vec,'*',label = "$\\mu_p$, LDP")
    plt.plot(eos.rho_apr03_vec,eos.mu_p_HDP_apr03_vec,'*',label = "$\\mu_p$, HDP")
    plt.plot(eos.rho_apr03_combined_vec,eos.mu_n_apr03_combined_vec,label = "$\\mu_n$, APR03")
    plt.plot(eos.rho_apr03_combined_vec,eos.mu_p_apr03_combined_vec,label = "$\\mu_p$, APR03")
    plt.legend()
    plt.savefig("figures/tests/figures_xeos_note/mu_i_of_rho.pdf")
    plt.show()

if(testing_CFL==True):
    time_0 = time.perf_counter()
    N_CFL = 3000
    N_CFL_kaons = 3000
    B = 190
    c = 0.
    Delta = 100
    m_s = 150
    N_apr03 = 1000
    eos = EoS_classes.CFL_EoS(N_CFL,N_CFL_kaons,B,Delta,m_s,N_apr03=N_apr03,c=c)
    print("Time spent:", time.perf_counter()-time_0)

    plt.figure()
    plt.xlim(300,425)
    plt.ylim(0,225)
    plt.xlabel("$\\mu_q$[MeV]")
    plt.ylabel("P[MeV/fm$^3$] / $\\mu[MeV]$")
    plt.plot(eos.mu_q_CFL_vec,eos.P_CFL_vec,label = "$P_{CFL}$",color = "green")
    plt.plot(eos.mu_q_CFL_with_kaons_vec,eos.P_CFL_with_kaons_vec,label = "$P_{CFL}^k$",color="purple")
    plt.plot(eos.mu_q_CFL_with_kaons_vec,eos.mu_e_CFL_with_kaons_vec,'*',label = "$\\mu_e^k$",color="red")
    plt.plot(eos.eos_apr03.mu_n_apr03_combined_vec/3,eos.eos_apr03.mu_e_apr03_combined_vec,label = "$\\mu_e$",color="black")
    plt.plot(eos.eos_apr03.mu_n_apr03_combined_vec/3,eos.eos_apr03.P_apr03_combined_vec,label = "$P_{NM}$",color="blue")
    plt.plot(eos.mu_q_vec,eos.P_vec,'--',label = "P_vec",color="red")
    plt.legend()
    plt.show()
    plt.savefig("figures/tests/figures_xeos_note/phases.pdf")
    
if(testing_total_eos_B==True):
    
    time_0 = time.perf_counter()
    
    N_B = 10
    
    B_vec = np.linspace(175,210,N_B)
    
    #First create apr eos, so we dont have to call it repeatedly
    N_apr03 = 2000
    eos_apr03 = EoS_classes.apr03_EoS(N_apr03,rho_min_apr03=0.1,rho_max_apr03=2)
   
    N_CFL = 2000
    N_CFL_kaons = 2000
    Delta = 100
    m_s = 150
    c = 0.3

    plt.figure(1)
    plt.xlim(320,480)
    plt.ylim(0,400)
    plt.title("$\\Delta$ = "+str(Delta)+", $m_s$ = "+str(m_s)+", $c$ = "+str(c))
    plt.xlabel("$\\mu_q$[MeV]")
    plt.ylabel("$P$[MeV/fm$^3$]")

    plt.figure(2)
    plt.xlim(0,1000)
    plt.ylim(0,400)
    plt.title("$\\Delta$ = "+str(Delta)+", $m_s$ = "+str(m_s)+", $c$ = "+str(c))
    plt.xlabel("$\\epsilon$[MeV/fm$^3$]")
    plt.ylabel("P[MeV/fm$^3$]")

    plt.figure(3)
    plt.xlim(0,2)
    plt.ylim(0,400)
    plt.title("$\\Delta$ = "+str(Delta)+", $m_s$ = "+str(m_s)+", $c$ = "+str(c))
    plt.xlabel("$\\rho$[fm$^{-3}$]")
    plt.ylabel("P[MeV/fm$^3$]")


    for i in range(N_B):
        B = B_vec[i]
        eos = EoS_classes.CFL_EoS(N_CFL,N_CFL_kaons,B,Delta,m_s,eos_apr03=eos_apr03,c=c)
        
        #Remove zeros from vectors
        mu_q_vec = eos.mu_q_vec[eos.mu_q_vec==0] = np.nan
        P_vec = eos.P_vec[eos.P_vec==0] = np.nan
        e_vec = eos.e_vec[eos.e_vec==0] = np.nan
        ##

        plt.figure(1) 
        plt.plot(eos.mu_q_vec,eos.P_vec,label = "B$^{1/4}$="+str(np.round(B,decimals=1))+"MeV")
        plt.legend()
        
        plt.figure(2)
        plt.plot(eos.e_vec,eos.P_vec,label = "B$^{1/4}$="+str(np.round(B,decimals=1))+"MeV")
        plt.legend()

        plt.figure(3)
        plt.plot(eos.rho_vec,eos.P_vec,label = "B$^{1/4}$="+str(np.round(B,decimals=1))+"MeV")
        plt.legend()
    
    plt.figure(1)
    plt.plot(eos.eos_apr03.mu_n_apr03_combined_vec/3,eos.eos_apr03.P_apr03_combined_vec,label = "NM",color = "black")
    plt.legend()

    plt.figure(2)
    plt.plot(eos.eos_apr03.e_apr03_combined_vec,eos.eos_apr03.P_apr03_combined_vec,label = "NM",color = "black")
    plt.legend()
    
    plt.figure(3)
    plt.plot(eos.eos_apr03.rho_apr03_combined_vec,eos.eos_apr03.P_apr03_combined_vec,label = "NM",color = "black")
    plt.legend()

    print("Time spent:", time.perf_counter()-time_0)
    plt.figure(1)
    plt.savefig("figures/tests/parameter_varying/B/P_of_mu_q.pdf")
    plt.figure(2)
    plt.savefig("figures/tests/parameter_varying/B/P_of_e.pdf")
    plt.figure(3)
    plt.savefig("figures/tests/parameter_varying/B/P_of_rho.pdf")

    plt.show()

if(testing_total_eos_Delta==True):
    
    time_0 = time.perf_counter()
    
    N_Delta = 10
    
    Delta_vec = np.linspace(50,150,N_Delta)
    
    #First create apr eos, so we dont have to call it repeatedly
    N_apr03 = 2000
    eos_apr03 = EoS_classes.apr03_EoS(N_apr03,rho_min_apr03=0.1,rho_max_apr03=2)
   
    N_CFL = 2000
    N_CFL_kaons = 2000
    B = 190
    m_s = 150
    c = 0.3

    plt.figure(1)
    plt.xlim(320,480)
    plt.ylim(0,400)
    plt.title("$m_s$ = "+str(m_s)+", $B$ = "+str(B)+", $c$ = "+str(c))
    plt.xlabel("$\\mu_q$[MeV]")
    plt.ylabel("$P$[MeV/fm$^3$]")

    plt.figure(2)
    plt.xlim(0,1000)
    plt.ylim(0,400)
    plt.title("$m_s$ = "+str(m_s)+", $B$ = "+str(B)+", $c$ = "+str(c))
    plt.xlabel("$\\epsilon$[MeV/fm$^3$]")
    plt.ylabel("P[MeV/fm$^3$]")

    plt.figure(3)
    plt.xlim(0,2)
    plt.ylim(0,400)
    plt.title("$m_s$ = "+str(m_s)+", $B$ = "+str(B)+", $c$ = "+str(c))
    plt.xlabel("$\\rho$[fm$^{-3}$]")
    plt.ylabel("P[MeV/fm$^3$]")


    for i in range(N_Delta):
        Delta = Delta_vec[i]
        eos = EoS_classes.CFL_EoS(N_CFL,N_CFL_kaons,B,Delta,m_s,eos_apr03=eos_apr03,c=c)
        
        #Remove zeros from vectors
        mu_q_vec = eos.mu_q_vec[eos.mu_q_vec==0] = np.nan
        P_vec = eos.P_vec[eos.P_vec==0] = np.nan
        e_vec = eos.e_vec[eos.e_vec==0] = np.nan
        ##

        plt.figure(1) 
        plt.plot(eos.mu_q_vec,eos.P_vec,label = "$\\Delta$="+str(np.round(Delta,decimals=1))+"MeV")
        plt.legend()
        
        plt.figure(2)
        plt.plot(eos.e_vec,eos.P_vec,label = "$\\Delta$="+str(np.round(Delta,decimals=1))+"MeV")
        plt.legend()

        plt.figure(3)
        plt.plot(eos.rho_vec,eos.P_vec,label = "$\\Delta$="+str(np.round(Delta,decimals=1))+"MeV")
        plt.legend()
    
    plt.figure(1)
    plt.plot(eos.eos_apr03.mu_n_apr03_combined_vec/3,eos.eos_apr03.P_apr03_combined_vec,label = "NM",color = "black")
    plt.legend()

    plt.figure(2)
    plt.plot(eos.eos_apr03.e_apr03_combined_vec,eos.eos_apr03.P_apr03_combined_vec,label = "NM",color = "black")
    plt.legend()
    
    plt.figure(3)
    plt.plot(eos.eos_apr03.rho_apr03_combined_vec,eos.eos_apr03.P_apr03_combined_vec,label = "NM",color = "black")
    plt.legend()

    print("Time spent:", time.perf_counter()-time_0)
    plt.figure(1)
    plt.savefig("figures/tests/parameter_varying/Delta/P_of_mu_q.pdf")
    plt.figure(2)
    plt.savefig("figures/tests/parameter_varying/Delta/P_of_e.pdf")
    plt.figure(3)
    plt.savefig("figures/tests/parameter_varying/Delta/P_of_rho.pdf")

    plt.show()

if(testing_total_eos_m_s==True):
    
    time_0 = time.perf_counter()
    
    N_m_s = 10
    
    m_s_vec = np.linspace(50,250,N_m_s)
    
    #First create apr eos, so we dont have to call it repeatedly
    N_apr03 = 2000
    eos_apr03 = EoS_classes.apr03_EoS(N_apr03,rho_min_apr03=0.1,rho_max_apr03=2)
   
    N_CFL = 2000
    N_CFL_kaons = 2000
    B = 190
    Delta = 100
    c = 0.3

    plt.figure(1)
    plt.xlim(320,480)
    plt.ylim(0,400)
    plt.title("$\\Delta$ = "+str(Delta)+", $B$ = "+str(B)+", $c$ = "+str(c))
    plt.xlabel("$\\mu_q$[MeV]")
    plt.ylabel("$P$[MeV/fm$^3$]")

    plt.figure(2)
    plt.xlim(0,1000)
    plt.ylim(0,400)
    plt.title("$\\Delta$ = "+str(Delta)+", $B$ = "+str(B)+", $c$ = "+str(c))
    plt.xlabel("$\\epsilon$[MeV/fm$^3$]")
    plt.ylabel("P[MeV/fm$^3$]")

    plt.figure(3)
    plt.xlim(0,2)
    plt.ylim(0,400)
    plt.title("$\\Delta$ = "+str(Delta)+", $B$ = "+str(B)+", $c$ = "+str(c))
    plt.xlabel("$\\rho$[fm$^{-3}$]")
    plt.ylabel("P[MeV/fm$^3$]")


    for i in range(N_m_s):
        m_s = m_s_vec[i]
        eos = EoS_classes.CFL_EoS(N_CFL,N_CFL_kaons,B,Delta,m_s,eos_apr03=eos_apr03,c=c)
        
        #Remove zeros from vectors
        mu_q_vec = eos.mu_q_vec[eos.mu_q_vec==0] = np.nan
        P_vec = eos.P_vec[eos.P_vec==0] = np.nan
        e_vec = eos.e_vec[eos.e_vec==0] = np.nan
        ##

        plt.figure(1) 
        plt.plot(eos.mu_q_vec,eos.P_vec,label = "$m_s$="+str(np.round(m_s,decimals=1))+"MeV")
        plt.legend()
        
        plt.figure(2)
        plt.plot(eos.e_vec,eos.P_vec,label = "$m_s$="+str(np.round(m_s,decimals=1))+"MeV")
        plt.legend()

        plt.figure(3)
        plt.plot(eos.rho_vec,eos.P_vec,label = "$m_s$="+str(np.round(m_s,decimals=1))+"MeV")
        plt.legend()
    
    plt.figure(1)
    plt.plot(eos.eos_apr03.mu_n_apr03_combined_vec/3,eos.eos_apr03.P_apr03_combined_vec,label = "NM",color = "black")
    plt.legend()

    plt.figure(2)
    plt.plot(eos.eos_apr03.e_apr03_combined_vec,eos.eos_apr03.P_apr03_combined_vec,label = "NM",color = "black")
    plt.legend()
    
    plt.figure(3)
    plt.plot(eos.eos_apr03.rho_apr03_combined_vec,eos.eos_apr03.P_apr03_combined_vec,label = "NM",color = "black")
    plt.legend()

    print("Time spent:", time.perf_counter()-time_0)
    plt.figure(1)
    plt.savefig("figures/tests/parameter_varying/m_s/P_of_mu_q.pdf")
    plt.figure(2)
    plt.savefig("figures/tests/parameter_varying/m_s/P_of_e.pdf")
    plt.figure(3)
    plt.savefig("figures/tests/parameter_varying/m_s/P_of_rho.pdf")

    plt.show()
