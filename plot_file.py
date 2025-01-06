import write_eos_to_file as wrt
import matplotlib.pyplot as plt

filename_NM = "runs/tests/no_mix_vs_mix/RMF_MR_0"
filename_NO_MIX = "runs/tests/test_no_mix_MR_0"
filename_KAONS = "runs/tests/test_kaons_MR_0"

M_vec,R_vec,Lambda_vec,P_c_vec,rho_c_vec = wrt.read_MR_from_file(filename_NM)
plt.plot(R_vec,M_vec,label="RMF")
M_vec,R_vec,Lambda_vec,P_c_vec,rho_c_vec = wrt.read_MR_from_file(filename_NO_MIX)
plt.plot(R_vec,M_vec,label="no mix")
M_vec,R_vec,Lambda_vec,P_c_vec,rho_c_vec = wrt.read_MR_from_file(filename_KAONS)
plt.plot(R_vec,M_vec,label="kaons")
plt.legend()
plt.show()
pltsavefig("runs/tests/no_mix_vs_mix/check_MR")
