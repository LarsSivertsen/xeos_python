import APR03_eos
import matplotlib.pyplot as plt
import time
import numpy as np
import write_eos_to_file as wrt
import TOV_Rahul
import exotic_eos
import RMF_eos
import create_random_parameters as crp
import write_eos_to_file as wrt



def testing_apr03(N=1000,figsize=(10,10)):
    '''
    Test attributes of the APR03 phase
    '''
    time_0 = time.perf_counter()

    eos = APR03_eos.apr03_EoS(N=N)
    print("Time spent:", time.perf_counter()-time_0)

    plt.figure(figsize=figsize)
    plt.xlim(0.1,1)
    plt.ylim(-50,450)
    plt.xlabel("$\\rho$ [fm$^{-3}$]")
    plt.ylabel("E/A [MeV]")

    plt.plot(eos.rho_PNM_SNM_vec,eos.E_per_A_PNM_LDP_vec,label = "PNM, LDP",color="Black")
    plt.plot(eos.rho_PNM_SNM_vec,eos.E_per_A_PNM_HDP_vec,label = "PNM, HDP",color = "Red")
    plt.plot(eos.rho_PNM_SNM_vec,eos.E_per_A_SNM_LDP_vec,'--',label = "SNM, LDP", color = "Black")
    plt.plot(eos.rho_PNM_SNM_vec,eos.E_per_A_SNM_HDP_vec,'--',label = "SNM, HDP", color = "Red")
    plt.legend()
    plt.savefig("figures/tests/figures_xeos_note/E_per_A_PNM_and_SNM.pdf")

    plt.figure(figsize=figsize)
    plt.xlim(0.1,1)
    plt.ylim(-50,450)
    plt.xlabel("$\\rho$ [fm$^{-3}$]")
    plt.ylabel("E/A [MeV]")
    plt.plot(eos.rho_LDP_HDP_vec,eos.E_per_A_LDP_vec,label = "LDP",color = "Black")
    plt.plot(eos.rho_LDP_HDP_vec,eos.E_per_A_HDP_vec,label = "HDP",color = "Red")
    plt.legend()
    plt.savefig("figures/tests/figures_xeos_note/E_per_A_beta_equillibrium.pdf")

    plt.figure(figsize=figsize)
    plt.xlim(0.1,1.2)
    plt.ylim(0,0.2)
    plt.xlabel("$\\rho$ [fm$^{-3}$]")
    plt.ylabel("$x_p$")
    plt.plot(eos.rho_LDP_HDP_vec,eos.xp_LDP_vec,'*',label = "LDP", color = "Blue")
    plt.plot(eos.rho_LDP_HDP_vec,eos.xp_HDP_vec,'*',label = "HDP", color = "Red")
    plt.plot(eos.rho_vec,eos.xp_vec,label = "APR03", color="Black")
    plt.legend()
    plt.savefig("figures/tests/figures_xeos_note/proton_fraction.pdf")

    plt.figure(figsize=figsize)
    plt.xlim(0.1,1)
    plt.ylim(0,600)
    plt.xlabel("$\\rho$ [fm$^{-3}$]")
    plt.ylabel("Energy density [MeV/fm$^3$]")

    plt.plot(eos.rho_LDP_HDP_vec,eos.e_LDP_vec,'*',color = "Blue",label = "LDP")
    plt.plot(eos.rho_LDP_HDP_vec,eos.e_HDP_vec,'*',color = "Red",label = "HDP")
    plt.plot(eos.rho_vec,eos.e_vec,color = "Black",label = "APR03")
    plt.legend()
    plt.savefig("figures/tests/figures_xeos_note/e_of_rho.pdf")

    plt.figure(figsize=figsize)
    plt.xlim(0.1,100)
    plt.ylim(0,600)
    plt.xlabel("Pressure [MeV/fm$^3$]")
    plt.ylabel("Energy density [MeV/fm$^3$]")
    plt.plot(eos.P_LDP_vec,eos.e_LDP_vec,'*',label = "LDP",color = "Blue")
    plt.plot(eos.P_HDP_vec,eos.e_HDP_vec,'*',label = "HDP",color = "Red")
    plt.plot(eos.P_vec,eos.e_vec,label = "APR03", color = "Black")
    plt.legend()
    plt.savefig("figures/tests/figures_xeos_note/e_of_P.pdf")

    plt.figure(figsize=figsize)
    plt.xlim(0.1,1)
    plt.ylim(800,2100)
    plt.xlabel("$\\rho$ [fm$^{-3}$]")
    plt.ylabel("$\\mu$ [MeV]")
    plt.plot(eos.rho_LDP_HDP_vec,eos.mu_n_LDP_vec,'*',label = "$\\mu_n$, LDP")
    plt.plot(eos.rho_LDP_HDP_vec,eos.mu_n_HDP_vec,'*',label = "$\\mu_n$, HDP")
    plt.plot(eos.rho_LDP_HDP_vec,eos.mu_p_LDP_vec,'*',label = "$\\mu_p$, LDP")
    plt.plot(eos.rho_LDP_HDP_vec,eos.mu_p_HDP_vec,'*',label = "$\\mu_p$, HDP")
    plt.plot(eos.rho_vec,eos.mu_n_vec,label = "$\\mu_n$, APR03")
    plt.plot(eos.rho_vec,eos.mu_p_vec,label = "$\\mu_p$, APR03")
    plt.legend()
    plt.savefig("figures/tests/figures_xeos_note/mu_i_of_rho.pdf")
    plt.show()

def testing_CFL(B=190
                ,Delta=100
                ,m_s=150
                ,c=0.0
                ,N=100
                ,N_kaons=300
                ,N_low_dens=300
                ,eos_name="APR"
                ,plot_original_phases=False
                ,RMF_filename="FSUGarnet.inp"
                ,mix_phase=True
                ,figsize=(10,10)):
    '''
    Test the CFL phase in combination with differnt low density phases.
    When the flag plot_original_phases is set to True, we plot results
    using a simple Waleca model (Marks code)
    '''
    time_0 = time.perf_counter()

    plt.figure(figsize=figsize)
    if(mix_phase==False):
        title = "No mixed phase. "
    else:
        title=""
    title += ("Low density eos: "+eos_name+"\n B="
             +str(np.round(B,2))+"(MeV)$^{1/4}$"+", $\\Delta=$"
             +str(np.round(Delta))+"MeV, $m_s$="+str(np.round(m_s,2))
             +"MeV, c="+str(np.round(c,2))
             )
    if(plot_original_phases==True):
        eos = exotic_eos.CFL_EoS(B
                                 ,Delta
                                 ,m_s
                                 ,N=N
                                 ,N_kaons=N_kaons
                                 ,N_low_dens=N_low_dens
                                 ,c=c
                                 ,eos_name=eos_name
                                 ,RMF_filename="FSUGold_MARK.inp"
                                 ,mix_phase=mix_phase)

        print("Time spent:", time.perf_counter()-time_0)
        title="Waleca parameters, "+title
        textfile = open('pmqk.dat')
        textfile_0 = open('pmqk_c=0.dat')
        mu_q_0 = []
        mu_e_k_0 = []
        p_0 = []

        mu_q = []
        mu_e_k = []
        p = []


        for line in textfile_0:
            data = line.split()
            if(len(data)>0):
                mu_q_0.append(float(data[0]))
                mu_e_k_0.append(float(data[1]))
                p_0.append(float(data[2]))

        for line in textfile:
            data = line.split()
            if(len(data)>0):
                mu_q.append(float(data[0]))
                mu_e_k.append(float(data[1]))
                p.append(float(data[2]))
        textfile.close()
        textfile_0.close()
        n_stop = 2100
        plt.plot(mu_q[:n_stop],p[:n_stop],"Magenta",label = "P Walec")
        plt.plot(mu_q,mu_e_k,label = "$\\mu_e$ Walec")
        plt.plot(mu_q_0,mu_e_k_0,label = "$\\mu_e$ Walec c=0")
        original = "_with_Mark"
    else:
        original = ""
        eos = exotic_eos.CFL_EoS(B,Delta,m_s,N=N,N_kaons=N_kaons,N_low_dens=N_low_dens,c=c,eos_name=eos_name,RMF_filename=RMF_filename,mix_phase=mix_phase)
        print("Time spent:", time.perf_counter()-time_0)

    plt.title(title)
    plt.xlim(300,530)
    plt.ylim(0,300)
    plt.xlabel("$\\mu_q$[MeV]")
    plt.ylabel("P[MeV/fm$^3$] / $\\mu[MeV]$")
    plt.plot(eos.mu_q_CFL_vec,eos.P_CFL_vec,label = "$P_{CFL}$",color = "green")
    if(mix_phase==True):
        plt.plot(eos.mu_q_CFL_with_kaons_vec,eos.P_CFL_with_kaons_vec,label = "$P_{CFL}^k$",color="orange")
        plt.plot(eos.mu_q_CFL_with_kaons_vec,eos.mu_e_CFL_with_kaons_vec,'--',label = "$\\mu_e^k$",color="black")
    plt.plot(eos.eos_low_dens.mu_n_vec/3,eos.eos_low_dens.mu_e_vec,label = "$\\mu_e$",color="black")
    plt.plot(eos.eos_low_dens.mu_n_vec/3,eos.eos_low_dens.P_vec,label = "$P_{NM}$",color="blue")
    plt.plot(eos.mu_q_vec,eos.P_vec,'--',label = "P",color="red")



    plt.legend()
    quark_interact="_c=0"+str(int(c*10))
    if(plot_original_phases==True):
        original = "_with_mark"
    else:

        original=""
    if(mix_phase==False):
        mix = "_no_mix"
    else:
        mix = ""
    plt.savefig("figures/tests/figures_xeos_note/phases"+mix+"_"+eos_name+quark_interact+original+".pdf")
    plt.show()

def testing_total_eos(B=190
                     ,Delta=100
                     ,m_s=150
                     ,c=0.0
                     ,variable="B"
                     ,variable_range=[170,210]
                     ,N_variable=5
                     ,N=100
                     ,N_kaons=300
                     ,N_low_dens=100
                     ,rho_max=3
                     ,eos_name="APR"
                     ,RMF_filename="FSUGarnet.inp"
                     ,mix_phase=True
                     ,figsize=(10,10)
                     ,plot_low_dens=True
                     ,skip_repeating_low_dens=True
                     ,TOV=False):


    time_0 = time.perf_counter()
    variable_names = ["B","Delta","m_s","c"]
    if(skip_repeating_low_dens):
        eos_low_dens = RMF_eos.EOS_set(N=N_low_dens,rho_max=rho_max,RMF_filename=RMF_filename,TOV=TOV)
    else:
        eos_low_dens=None
    var_vec = np.linspace(variable_range[0],variable_range[1],N_variable)
    if(variable=="B"):
        rnd=1
        title = "$\\Delta=$"+str(np.round(Delta,1))+"MeV, $m_s$="+str(np.round(m_s,1))+"MeV, c="+str(np.round(c,3))
        label_variable="B"
        unit = "(MeV)$^{1/4}$"
        B_vec = np.linspace(variable_range[0],variable_range[1],N_variable)
        all_eos = [exotic_eos.CFL_EoS(B
                                  ,Delta
                                  ,m_s
                                  ,N=N
                                  ,N_kaons=N_kaons
                                  ,N_low_dens=N_low_dens
                                  ,c=c
                                  ,eos_name=eos_name
                                  ,RMF_filename=RMF_filename
                                  ,mix_phase=mix_phase
                                  ,eos_low_dens=eos_low_dens
                                  ,TOV=TOV)
               for B in var_vec
               ]

    elif(variable=="Delta"):
        rnd=1
        title = "B$^{1/4}$="+str(np.round(B,1))+"MeV, $m_s$="+str(np.round(m_s,1))+"MeV, c="+str(np.round(c,3))
        label_variable="$\\Delta$"
        unit = "MeV"
        all_eos = [exotic_eos.CFL_EoS(B
                                  ,Delta
                                  ,m_s
                                  ,N=N
                                  ,N_kaons=N_kaons
                                  ,N_low_dens=N_low_dens
                                  ,c=c
                                  ,eos_name=eos_name
                                  ,RMF_filename=RMF_filename
                                  ,mix_phase=mix_phase
                                  ,eos_low_dens=eos_low_dens
                                  ,TOV=TOV)
               for Delta in var_vec
               ]
    elif(variable=="m_s"):
        rnd=1
        title = "B$^{1/4}$="+str(np.round(B,1))+"MeV, $\\Delta$="+str(np.round(Delta,1))+"MeV, c="+str(np.round(c,3))
        label_variable="$m_s$"
        unit = "MeV"
        all_eos = [exotic_eos.CFL_EoS(B
                                  ,Delta
                                  ,m_s
                                  ,N=N
                                  ,N_kaons=N_kaons
                                  ,N_low_dens=N_low_dens
                                  ,c=c
                                  ,eos_name=eos_name
                                  ,RMF_filename=RMF_filename
                                  ,mix_phase=mix_phase
                                  ,eos_low_dens=eos_low_dens
                                  ,TOV=TOV)
               for m_s in var_vec
               ]
    elif(variable=="c"):
        rnd=3
        title = "B$^{1/4}$="+str(np.round(B,1))+"MeV, $\\Delta$="+str(np.round(Delta,1))+"MeV"
        label_variable="c"
        unit = ""
        all_eos = [exotic_eos.CFL_EoS(B
                                  ,Delta
                                  ,m_s
                                  ,N=N
                                  ,N_kaons=N_kaons
                                  ,N_low_dens=N_low_dens
                                  ,c=c
                                  ,eos_name=eos_name
                                  ,RMF_filename=RMF_filename
                                  ,mix_phase=mix_phase
                                  ,eos_low_dens=eos_low_dens
                                  ,TOV=TOV)
               for c in var_vec
               ]
    else:
        print("Invalid variable")
        return

    print("Time spent:", time.perf_counter()-time_0)
    plt.figure(figsize=figsize)
    plt.title("Pressure as a function of energy density\n"+title)

    for var,eos in zip(var_vec,all_eos):
        eos.add_crust()
        plt.plot(eos.e_vec,eos.P_vec,label=label_variable+"="+str(np.round(var,rnd))+unit)

    plt.plot(all_eos[0].eos_low_dens.e_vec,all_eos[0].eos_low_dens.P_vec,"k--",label=eos_name)
    plt.xlabel("$\\epsilon$[MeV/fm$^3$]")
    plt.ylabel("P[MeV/fm$^3$]")
    plt.xlim(0,1300)
    plt.ylim(0,200)
    plt.legend()
    plt.savefig("figures/tests/parameter_varying/"+variable+"/"+eos_name+"_P_of_e"+".pdf")

    plt.figure(figsize=figsize)
    plt.title("Pressure as a function of density\n"+title)

    for var,eos in zip(var_vec,all_eos):
        plt.plot(eos.rho_vec,eos.P_vec,label=label_variable+"="+str(np.round(var,rnd))+unit)

    plt.plot(all_eos[0].eos_low_dens.rho_vec,all_eos[0].eos_low_dens.P_vec,"k--",label=eos_name)
    plt.xlabel("$\\rho$[fm$^{-3}$]")
    plt.ylabel("P[MeV/fm$^3$]")
    plt.xlim(0.1,rho_max)
    plt.ylim(0,200)
    plt.legend()
    plt.savefig("figures/tests/parameter_varying/"+variable+"/"+eos_name+"_P_of_rho"+".pdf")

    plt.figure(figsize=figsize)
    plt.title("Pressure as a function of quark chemical potential\n"+title)
    for var,eos in zip(var_vec,all_eos):
        plt.plot(eos.mu_q_vec,eos.P_vec,label=label_variable+"="+str(np.round(var,rnd))+unit)

    plt.plot(all_eos[0].eos_low_dens.mu_n_vec/3,all_eos[0].eos_low_dens.P_vec,"k--",label=eos_name)
    plt.xlabel("$\\mu_q$[MeV]")
    plt.ylabel("P[MeV/fm$^3$]")
    plt.xlim(312,470)
    plt.ylim(0,200)
    plt.legend()
    plt.savefig("figures/tests/parameter_varying/"+variable+"/"+eos_name+"_P_of_mu_q"+".pdf")

    plt.figure(figsize=figsize)
    plt.title("Speed of sound as a function of density\n"+title)
    for var,eos in zip(var_vec,all_eos):
        plt.plot(eos.rho_vec,eos.v2_vec,label=label_variable+"="+str(np.round(var,rnd))+unit)

    plt.plot(all_eos[0].eos_low_dens.rho_vec,all_eos[0].eos_low_dens.v2_vec,"k--",label=eos_name)
    plt.xlabel("$\\rho$[fm$^{-3}$]")
    plt.ylabel("$v_s^2$")
    plt.xlim(0.,3)
    plt.ylim(0,1.2)
    plt.legend()
    plt.savefig("figures/tests/parameter_varying/"+variable+"/"+eos_name+"_v2_of_rho"+".pdf")

    if(TOV==True):
        plt.figure(figsize=figsize)
        plt.title("Mass-radius plot\n"+title)
        for var,eos in zip(var_vec,all_eos):
            plt.plot(eos.R_vec,eos.M_vec,label=label_variable+"="+str(np.round(var,rnd))+unit)

        plt.plot(all_eos[0].eos_low_dens.R_vec,all_eos[0].eos_low_dens.M_vec,"k--",label=eos_name)
        plt.xlabel("R[km]")
        plt.ylabel("M[$M_\\odot$]")
        plt.xlim(8,15)
        plt.ylim(0,2.2)
        plt.legend()
        plt.savefig("figures/tests/parameter_varying/"+variable+"/"+eos_name+"_M_of_R"+".pdf")

        plt.figure(figsize=figsize)
        plt.title("Mass as function of central pressure\n"+title)
        for var,eos in zip(var_vec,all_eos):
            plt.plot(eos.P_c_vec,eos.M_vec,label=label_variable+"="+str(np.round(var,rnd))+unit)

        plt.plot(all_eos[0].eos_low_dens.P_c_vec,all_eos[0].eos_low_dens.M_vec,"k--",label=eos_name)
        plt.xlabel("$P_c$[MeV/fm$^3$]")
        plt.ylabel("M[$M_\\odot$]")
        plt.xlim(0,300)
        plt.ylim(0,2.2)
        plt.legend()
        plt.savefig("figures/tests/parameter_varying/"+variable+"/"+eos_name+"_M_of_P_c"+".pdf")

        plt.figure(figsize=figsize)
        plt.yscale("log")
        plt.title("Deformability as function of mass\n"+title)
        for var,eos in zip(var_vec,all_eos):
            plt.plot(eos.M_vec,eos.Lambda_vec,label=label_variable+"="+str(np.round(var,rnd))+unit)

        plt.plot(all_eos[0].eos_low_dens.M_vec,all_eos[0].eos_low_dens.Lambda_vec,"k--",label=eos_name)
        plt.xlabel("M[$M_\\odot$]")
        plt.ylabel("$\\Lambda$")
        plt.xlim(0.8,2.2)
        plt.ylim(0,4000)
        plt.legend()
        plt.savefig("figures/tests/parameter_varying/"+variable+"/"+eos_name+"_Lambda_of_M"+".pdf")

    plt.show()

def test_writing_parameter_sets(filename="tests/all/parameters.txt"
                                ,N=10000
                                ,parameter_ranges = [[30,60],[90,1000],[0,30],[0,0.6]]
                                ,distributions=["uniform","uniform","uniform","uniform"]
                                ):
    crp.make_parameter_file(filename,N,parameter_ranges,distributions=distributions)
    B_vec,Delta_vec,m_s_vec,c_vec = crp.read_parameters_from_file(filename)

    plt.figure()
    plt.xlabel("$B^{1/4}$[MeV]")
    plt.hist(B_vec,bins = 100)
    plt.figure()
    plt.hist(Delta_vec,bins = 100)
    plt.xlabel("$\\Delta$[MeV]")
    plt.figure()
    plt.hist(m_s_vec,bins = 100)
    plt.xlabel("$m_s$[MeV]")
    plt.figure()
    plt.hist(c_vec,bins=100)
    plt.xlabel("c")
    plt.show()

def test_write_and_read_eos_to_file(filename="runs/tests/EoS_files/test.txt",B=190,Delta=100,m_s=150,c=0.):
    eos=exotic_eos.CFL_EoS(B
                       ,Delta
                       ,m_s
                       ,c=c)

    wrt.write_EoS_to_file(eos,filename)
    P_vec,e_vec,rho_vec,v2_vec = wrt.read_EoS_from_file(filename)
    plt.figure()
    plt.plot(e_vec,P_vec)
    plt.xlabel("$\\epsilon$[MeV/fm$^{3}$]")
    plt.ylabel("P[MeV/fm$^{3}$]")

    plt.figure()
    plt.plot(rho_vec,v2_vec)
    plt.xlabel("$\\rho$[fm$^{-3}$]")
    plt.ylabel("$v_s^2$")

    plt.show()

def test_write_and_read_MR_to_file(filename="runs/tests/MR_files/test.txt",B=190,Delta=100,m_s=150,c=0.):
    eos=exotic_eos.CFL_EoS(B
                       ,Delta
                       ,m_s
                       ,c=c
                       ,TOV=True)

    wrt.write_MR_to_file(eos,filename)
    M_vec,R_vec,Lambda_vec,P_c_vec = wrt.read_MR_from_file(filename)
    plt.figure()
    plt.plot(R_vec,M_vec)
    plt.xlabel("R[km]]")
    plt.ylabel("M[$M_\\odot$]")

    plt.figure()
    plt.yscale("log")
    plt.plot(P_c_vec,Lambda_vec)
    plt.xlabel("$P_c$[MeV/fm$^3$]]")
    plt.ylabel("$\\Lambda$")

    plt.show()

