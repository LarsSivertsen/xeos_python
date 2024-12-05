#This file lists all the filenames in a folder and counts number of files
import os
import create_random_parameters as crm
import numpy as np

def make_filename_list(run_number,test):
    if(test):
        run_folder = "runs/test_runs/"
    else:
        run_folder = "runs/"

    arr = sorted(os.listdir(run_folder+'run_'+str(run_number)+'/EoS'))

    n = 0

    with open(run_folder+"run_"+str(run_number)+"/filenames.txt","w") as file:
        file.write(str(len(arr))+"\n")
        for name in arr:
            file.write(name+"\n")
            n+=1
    return n

def get_valid_parameters(run_number,test):
    if(test):
        run_folder = "runs/test_runs/"
    else:
        run_folder = "runs/"
    line_numbers = []
    with open(run_folder+"run_"+str(run_number)+"/filenames.txt",'r') as file:
        for line in file:
            try:
                line_number = int(file.readline()[:6])
                line_numbers.append(line_number)
            except:
                continue
    B_vec,Delta_vec,m_s_vec,c_vec = crm.read_parameters_from_file(run_folder+"run_"+str(run_number)+"/parameters.txt")

    B_vec_valid = []
    Delta_vec_valid = []
    m_s_vec_valid = []
    c_vec_valid = []

    for line_number in line_numbers:
        B_vec_valid.append(B_vec[line_number])
        Delta_vec_valid.append(Delta_vec[line_number])
        m_s_vec_valid.append(m_s_vec[line_number])
        c_vec_valid.append(c_vec[line_number])


    print("B min: "+str(min(B_vec_valid))+", B max: "+str(max(B_vec_valid)))
    print("Delta min: "+str(min(Delta_vec_valid))+", Delta max: "+str(max(Delta_vec_valid)))
    print("m_s min: "+str(min(m_s_vec_valid))+", m_s max: "+str(max(m_s_vec_valid)))
    print("c min: "+str(min(c_vec_valid))+", c max: "+str(max(c_vec_valid)))



    return np.array(B_vec_valid),np.array(Delta_vec_valid),np.array(m_s_vec_valid),np.array(c_vec_valid)

#get_valid_parameters(10,True)
