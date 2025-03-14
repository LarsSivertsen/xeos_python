import corner
import create_random_parameters as crp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

run_number = 3002
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

data = np.zeros((1000,4))
for i in range(1000):
    data[i,0] = B_vec[i]
    data[i,1] = Delta_vec[i]
    data[i,2] = m_s_vec[i]
    data[i,3] = c_vec[i]

df = pd.DataFrame({"$B$ (0,200)":data[:,0],"$\\Delta$ (0,1000)":data[:,1],"$m_s$ (80,120)":data[:,2],"$c$ (0,1.0)":data[:,3]})
plt.figure()

def envelope(x,a):
    return np.where(x<a,(x**2*6.4e-4-80*6.4e-4*x+315,2/5*x+100),(2/5*x+250.36,2/5*x+100))


#p = np.poly1d(np.polyfit(data[:,1],data[:,0],1))
x = np.linspace(0,1000,1000)
plt.xlim(0,500)
plt.ylim(0,500)
plt.plot(data[:,1],data[:,0],'.')
#plt.plot(x,p(x)+100)
#plt.plot(x,p(x)-70)
#plt.plot(x,x**2*6.4e-4-80*6.4e-4*x+315,"k")
#plt.plot(x,2/5*x+100,"k")
plt.plot(x,envelope(x,200)[0])
plt.plot(x,envelope(x,200)[1])
sns.pairplot(data=df)
plt.savefig("runs/run_"+str(run_number)+"/parameter_contour_plot.pdf")
plt.show()

