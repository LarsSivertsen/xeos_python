import corner
import create_random_parameters as crp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

run_number = 1012
B_vec,Delta_vec,m_s_vec,c_vec=crp.read_parameters_from_file('runs/run_'+str(run_number)+"/parameters.txt")
filenames_int = []
with open("runs/run_"+str(run_number)+"/filenames.txt","r") as all_filenames:
    n_filenames = int(all_filenames.readline())
    for filename in all_filenames:
        filenames_int.append(int(filename[:6]))
data = np.zeros((n_filenames,4))
j = 0
for i in filenames_int:
    data[j,0] = B_vec[i]
    data[j,1] = Delta_vec[i]
    data[j,2] = m_s_vec[i]
    data[j,3] = c_vec[i]
    j+=1

df = pd.DataFrame({"$B$ (0,200)":data[:,0],"$\\Delta$ (0,1000)":data[:,1],"$m_s$ (80,120)":data[:,2],"$c$ (0,1.0)":data[:,3]})
plt.figure()
sns.pairplot(data=df)
plt.savefig("runs/run_"+str(run_number)+"/parameter_contour_plot.pdf")
plt.show()

