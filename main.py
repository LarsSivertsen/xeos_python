import write_eos_to_file as wrt
import create_random_parameters as crp
import time
import numpy as np
import os
import exotic_eos
import RMF_eos
import APR03_eos
from multiprocessing import Pool, Process, cpu_count
import summarize_folder as sf
from tqdm import tqdm


def task(i,
         B_vec,
         Delta_vec,
         m_s_vec,
         c,
         eos_apr03,
         run_number,
         N_CFL,
         N_kaons,
         TOV,
         TOV_limit,
         run_folder):

    B = B_vec[i]
    Delta = Delta_vec[i]
    m_s = m_s_vec[i]
    c = c_vec[i]
    try:
        eos = exotic_eos.CFL_EoS(B,Delta,m_s,c=c,N=N_CFL,N_kaons=N_kaons,eos_low_dens=eos_low_dens,TOV=TOV,TOV_limit=TOV_limit,eos_name=eos_name)
        if(eos.status=="Success"):
            filename_eos = run_folder+"run_"+str(run_number)+"/EoS/"+str(i).zfill(6)+".txt"
            wrt.write_EoS_to_file(eos,filename_eos)
            filename_TOV = run_folder+"run_"+str(run_number)+"/TOV/"+str(i).zfill(6)+".txt"
            wrt.write_MR_to_file(eos,filename_TOV)
        else:
            return
    except:
        return




if __name__ == "__main__":

    time_0 = time.perf_counter()

    run_number = 1010
    test = False

    if(test):
        run_folder = "runs/test_runs/"
    else:
        run_folder = "runs/"
    if(os.path.isdir(run_folder+"run_"+str(run_number))==False):
        os.mkdir(run_folder+"run_"+str(run_number))
    TOV = True
    TOV_limit=True
    N_low_dens = 100
    rho_max = 1.5
    #RMF_filename="FSUGarnet.inp"
    #RMF_filename = "NL3.inp"
    RMF_filename = "FSUGarnet.inp"
    eos_name = "RMF"
    if(eos_name=="RMF"):
        eos_low_dens = RMF_eos.EOS_set(N=N_low_dens,rho_max=rho_max,RMF_filename=RMF_filename,TOV=TOV,TOV_limit=TOV_limit)
    elif(eos_name=="APR"):
        eos_low_dens = APR03_eos(N=N_low_dens)


    filename_par = run_folder+"run_"+str(run_number)+"/parameters.txt"

    N = 50000 #Number of EoS we compute
    N_CFL = 100
    N_kaons = 100


    B_range = [0,200]
    Delta_range = [0,1000]
    m_s_range = [80,120]
    c_range = [0.,1.0]

    parameter_ranges = [B_range,Delta_range,m_s_range,c_range]
    distributions=["uniform","uniform","uniform","uniform"]
    crp.make_parameter_file(filename_par,N,parameter_ranges,distributions=distributions)

    B_vec, Delta_vec, m_s_vec, c_vec = crp.read_parameters_from_file(filename_par)

    if(os.path.isdir(run_folder+"run_"+str(run_number)+"/EoS")==False):
        os.mkdir(run_folder+"run_"+str(run_number)+"/EoS")

    if(os.path.isdir(run_folder+"run_"+str(run_number)+"/TOV")==False):
        os.mkdir(run_folder+"run_"+str(run_number)+"/TOV")

    progress = 0
    d_progress = 100./N
    print("Done generating low density eos")
    print("Time spent generating low density eos:", time.perf_counter()-time_0)
    time_0 = time.perf_counter()
    print("Progress: 0%")

    j_N = min([100,N])
    k_N = int(np.floor(N/j_N))
    for k in tqdm(range(k_N)):
        for j in range(j_N):
            i = j+j_N*k
            #if(int(progress)<int(progress+d_progress)):
            #    print("progress: "+str(int(progress+d_progress))+"%")
            progress+=d_progress

            process = Process(target = task, args = (i,
                                                     B_vec,
                                                     Delta_vec,
                                                     m_s_vec,
                                                     c_vec,
                                                     eos_low_dens,
                                                     run_number,
                                                     N_CFL,
                                                     N_kaons,
                                                     TOV,
                                                     TOV_limit,
                                                     run_folder)
                              )
            process.start()
        process.join()
    print("Time spent on exotic part:", time.perf_counter()-time_0)
    num_valid = sf.make_filename_list(run_number,test)
    print(str(num_valid)+" or "+str(round(100*num_valid/N,1))+"% valid eos")









