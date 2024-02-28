import EoS_classes
import matplotlib.pyplot as plt
import time
import numpy as np

testing_apr03 = False#True
testing_CFL = True
testing_total_eos = True#True
testing_find_x_p_and_rho = False
mu_e_vec_test = False

time_0 = time.perf_counter()

#Testing
if(testing_apr03==True):

    
 
    N_apr03 = 100
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

    plt.figure()
    plt.xlim(0,1)
    plt.ylim(-50,450)
    plt.xlabel("$\\rho$ [fm$^{-3}$]")
    plt.ylabel("E/A [MeV]")
    plt.plot(eos.rho_apr03_vec,eos.E_per_A_LDP_apr03_vec,label = "LDP",color = "Black")
    plt.plot(eos.rho_apr03_vec,eos.E_per_A_HDP_apr03_vec,label = "HDP",color = "Red")
    plt.legend()

    plt.figure()
    plt.xlim(0,1.2)
    plt.ylim(0,0.2)
    plt.xlabel("$\\rho$ [fm$^{-3}$]")
    plt.ylabel("$x_p$")
    plt.plot(eos.rho_apr03_vec,eos.x_p_LDP_apr03_vec,'*',label = "LDP", color = "Blue")
    plt.plot(eos.rho_apr03_vec,eos.x_p_HDP_apr03_vec,'*',label = "HDP", color = "Red")
    plt.plot(eos.rho_apr03_combined_vec,eos.x_p_apr03_combined_vec,label = "APR03", color="Black")
    plt.legend()
   
    plt.figure()
    plt.xlim(0,1)
    plt.ylim(0,600)
    plt.xlabel("$\\rho_0$[fm$^{-3}$]")
    plt.ylabel("Energy density [MeV/fm$^3$]")
    
    plt.plot(eos.rho_apr03_vec,eos.e_LDP_apr03_vec,'*',color = "Blue",label = "LDP")
    plt.plot(eos.rho_apr03_vec,eos.e_HDP_apr03_vec,'*',color = "Red",label = "HDP")
    plt.plot(eos.rho_apr03_combined_vec,eos.e_apr03_combined_vec,color = "Black",label = "APR03")
    plt.legend()

    plt.figure()
    plt.xlim(0,100)
    plt.ylim(0,600)
    plt.xlabel("Pressure [MeV/fm$^3$]")
    plt.ylabel("Energy density [MeV/fm$^3$]")
    plt.plot(eos.P_LDP_apr03_vec,eos.e_LDP_apr03_vec,'*',label = "LDP",color = "Blue")
    plt.plot(eos.P_HDP_apr03_vec,eos.e_HDP_apr03_vec,'*',label = "HDP",color = "Red")
    plt.plot(eos.P_apr03_combined_vec,eos.e_apr03_combined_vec,label = "APR03", color = "Black")
    plt.legend()

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

    plt.show()

if(testing_CFL==True):
    N_CFL = 400
    N_CFL_kaons = 400
    B = 180
    Delta = 100
    m_s = 150
    N_apr03 = 100
    eos = EoS_classes.CFL_EoS(N_CFL,N_CFL_kaons,B,Delta,m_s,N_apr03=N_apr03)
    print("Time spent:", time.perf_counter()-time_0)

    plt.figure()
    plt.xlim(300,425)
    plt.ylim(0,225)
    plt.plot(eos.mu_q_CFL_vec,eos.P_CFL_vec,label = "$P_{CFL}$")
    plt.plot(eos.mu_q_CFL_vec,eos.P_CFL_test_vec,'--',label = "$P_{CFL}^{test}$")
    plt.plot(eos.mu_q_CFL_with_kaons_vec,eos.P_CFL_with_kaons_vec,'--',label = "$P_{CFL}^k$")
    plt.plot(eos.mu_q_CFL_with_kaons_vec,eos.mu_e_CFL_with_kaons_vec,'*',label = "$\\mu_e^k$")
    plt.plot(eos.eos_apr03.mu_n_apr03_combined_vec/3,eos.eos_apr03.mu_e_apr03_combined_vec,label = "$\\mu_e$")
    #plt.plot(eos.mu_q_CFL_vec,eos.mu_e_CFL_with_kaons_vec,'--',label = "$\\mu_e^{k, test}$")
    plt.plot(eos.eos_apr03.mu_n_apr03_combined_vec/3,eos.eos_apr03.P_apr03_combined_vec,label = "$P_{NM}$")
    plt.plot(eos.mu_q_vec,eos.P_of_mu_q_vec,label = "P_vec")
    #plt.plot(eos.mu_q_CFL_vec,eos.P_test,'--',label="Test")
    plt.legend()
    rho_vec = np.linspace(0,1,1000)
    plt.figure()
    plt.plot(rho_vec,eos.eos_apr03.LDP_HDP_apr03_intersect_P_functional(rho_vec))
    N = 100
    mu_q = 450
    pF = eos.fermi_momenta_CFL(mu_q,mu_q)
    mu_e = np.linspace(-500,500,N)
    check = np.zeros(N)
    check2 = np.zeros(N)
    check3 = np.zeros(N)
    check4 = np.zeros(N)
    for i in range(N):
        mu_n = 3*mu_q
        mu_p = mu_n-mu_e[i]
        kFN = np.sqrt(mu_n**2-eos.eos_apr03.m**2)
        if(mu_p>eos.eos_apr03.m):
            kFP = np.sqrt(mu_p**2-eos.eos_apr03.m**2)
        else:
            kFP = 0
        #print(mu_q,pF)
        #print(kFN,kFP)
        #kFN = np.sqrt(2*eos.eos_apr03.m*abs(mu_n-eos.eos_apr03.m))
        #kFP = np.sqrt(2*eos.eos_apr03.m*abs(mu_p-eos.eos_apr03.m))
        #print(kFN,kFP)
        #rho = (kFP**3+kFN**3)/(3*np.pi**2*eos.hc**3)
        #x_p = kFP**3/(kFN**3+kFP**3)
        #print("1)",rho,x_p,mu_n,mu_p,mu_e[i])
        #check[i] = eos.eos_apr03.total_pressure_of_mu_e_apr03(mu_n,mu_e[i],1)
        
        #check2[i] = eos.pressure_CFL_with_kaons(mu_q,pF,mu_e[i])
        #check3[i] = eos.pressure_CFL(mu_q,pF)
        #check4[i] = eos.chemical_potential_electron_CFL_with_kaons_functional(mu_e[i],pF[0],mu_q)
    #print(eos.chemical_potential_electron_CFL_with_kaons_functional(107,pF[0],mu_q))
    for i in range(len(eos.eos_apr03.mu_n_HDP_apr03_vec)):
        mu_n_val = eos.eos_apr03.mu_n_HDP_apr03_vec[i]
        mu_p_val = eos.eos_apr03.mu_p_HDP_apr03_vec[i]
        rho_val = eos.eos_apr03.rho_apr03_vec[i]
        x_p_val = eos.eos_apr03.x_p_HDP_apr03_vec[i]
        mu_e_val = eos.eos_apr03.mu_e_HDP_apr03_vec[i]
        #print("2)",rho_val,x_p_val,mu_n_val,mu_p_val,mu_e_val)
    plt.figure()
    plt.ylim(-200,200)
    #plt.plot(mu_e,check,label = "$P_{apr03}$")
    #plt.plot(mu_e,check2,label = "$P_{mix}$")
    #plt.plot(mu_e,check3,'--',label="$P_{CFL}$")
    #plt.plot(mu_e,check-check2,label="$P_{apr03}-P_{mix}$")
    plt.plot(mu_e,check4,label = "chemical_potential_electron_functional")
    
    plt.legend()
    #plt.figure()
    #plt.xlim(0,2)
    #plt.plot(eos.rho_CFL_vec,eos.P_CFL_vec)

    plt.show()


if(testing_total_eos==True):
    B_vec = np.linspace(140,210,5)
    N_apr03 = 1000
    eos_apr03 = EoS_classes.apr03_EoS(N_apr03,rho_min_apr03=0.1,rho_max_apr03=2)
    for i in range(len(B_vec)):
        plt.figure(1)
        plt.xlim(300,425)
        plt.ylim(0,225)
        plt.xlabel("$\\mu_q$[MeV]")
        plt.ylabel("$P$[MeV/fm$^3$]")

        N_CFL = 1000
        N_CFL_kaons = 1000
        Delta = 100
        m_s = 150
        eos = EoS_classes.CFL_EoS(N_CFL,N_CFL_kaons,B_vec[i],Delta,m_s,eos_apr03=eos_apr03)
        plt.plot(eos.mu_q_vec,eos.P_of_mu_q_vec,label = "B$^{1/4}$="+str(B_vec[i])+"MeV")
        plt.legend()
        
        plt.figure(2)
        plt.xlim(0,600)
        plt.ylim(0,100)
        plt.plot(eos.e_of_mu_q_vec,eos.P_of_mu_q_vec,'*')

        plt.figure(3)
        plt.xlim(0,2)
        plt.ylim(0,600)
        plt.plot(eos.rho_of_mu_q_vec,eos.P_of_mu_q_vec,'*')
    
    print("Time spent:", time.perf_counter()-time_0)
    plt.show()

if(testing_find_x_p_and_rho):
    plt.figure()
    N_apr03 = 100
    eos = EoS_classes.apr03_EoS(N_apr03)
    N = 100
    x_p_vec = np.linspace(-1,1,N)
    f1_vec = np.zeros(N)
    f2_vec = np.zeros(N)
    k = 40
    rho = eos.rho_apr03_combined_vec[k]
    mu_n = 972
    print(mu_n)
    mu_p = 764
    
    for i in range(N):
        f1_vec[i],f2_vec[i] = eos.find_x_p_and_rho_apr03_functional([x_p_vec[i],rho],mu_n,mu_p,1)
    
    #print(f1_vec)
    plt.plot(x_p_vec,f1_vec)
    plt.plot(x_p_vec,f2_vec)
    
    plt.figure()
    N_apr03 = 100
    eos = EoS_classes.apr03_EoS(N_apr03)
    N = 100
    x_p_vec = np.linspace(-1,1,N)
    f1_vec = np.zeros(N)
    f2_vec = np.zeros(N)
    k = 40
    rho_vec = np.linspace(-500,500,N)
    x_p = eos.x_p_apr03_combined_vec[k]
    print(mu_n)
    mu_e = 300
    mu_p = mu_n-mu_e
    #plt.ylim(-10000,10000)
    for i in range(N):
        f1_vec[i],f2_vec[i] = eos.find_x_p_and_rho_apr03_functional([x_p,rho_vec[i]],mu_n,mu_p,1)
    print(1-mu_p**3/mu_n**3,(mu_n**2-eos.m**2)**(3/2)/(3*np.pi**2*eos.hc**3))
    print(eos.find_x_p_and_rho_apr03(mu_n,mu_p,[1-mu_p**3/mu_n**3,(mu_n**2-eos.m**2)**(3/2)/(3*np.pi**2*eos.hc**3)],1))
    print(eos.find_x_p_and_rho_apr03(mu_n,mu_p,[0.1,-0.3],1))
    plt.plot(rho_vec,f1_vec)
    plt.plot(rho_vec,f2_vec)
    plt.show()

if(mu_e_vec_test==True):
    eos = EoS_classes.apr03_EoS(100)
    mu_n = 1000
    N = 100
    mu_e_vec = np.linspace(0,mu_n,N)
    f = np.zeros(N)
    for i in range(N):
        f[i] = eos.find_x_p_and_rho()

