import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import random
import TOV_Rahul
import RMF_eos as RMF
import APR03_eos as APR03
from scipy.interpolate import interp1d


#Class for computing the color flavored locked (CFL) phase (or just the exotic phase)
class CFL_EoS(object):
    def __init__(self,B,Delta,m_s               #Bag constant, pairng gap, strange quark mass
                     ,N_CFL = 100                   #Number of points used to compute CFL phase
                     ,N_low_dens = 100          #Number of points used in low density phase of the EoS                         #Bag constant
                     ,mu_q_min = 939/3.           #Minimum quark chemical potential we start computing CFL from [MeV]
                     ,mu_q_max = 1000.           #Maximum quark chemical potential we compute the CFL phase to [MeV]
                     ,rho_min_low_dens = .05   #Lowest number density we compute for the low density phaseAPR03 [fm^-3]
                     ,rho_max_low_dens = 1.5    #Max density we compute low density phase [fm^-3]
                     ,rho_max_high_dens = 6.4
                     ,d_rho = 0.0001            #When computing derivatives in APR, we use this step size in rho
                     ,tol = 1e-6                #Absolute tolerance used in root solvers finding electron density and proton fraction
                     ,c = 0.                    #Phenomenological parameter for quark interactions. Usually set to 0.3
                     ,eos_name = "APR"          #Name of nuclear EoS (supports at the moment APR and RMF)
                     ,TOV_limit = False         #True if we compute only up to dM/dR=0 in the TOV solver
                     ,eos_low_dens = None       #If we already have computed the low density EoS, we can insert it here, so we don't have to
                                                #compute it again
                     ,RMF_filename = "FSUgarnet.inp" #Filename for the parameters used in the RMF model
                     ,TOV=False                 #If set to True, compute MR curves in addition to EoS
                     ,mix_phase=True            #If set to False, there is no Kaon phase
                     ,filename_crust = "nveos.in"
                     ,dmu_q = 2
                     ,dmu_q_factor = 2
                 ):
        self.N_CFL = N_CFL
        self.B = B
        self.Delta = Delta
        self.m_s = m_s
        self.mu_q_min = mu_q_min
        self.mu_q_max = mu_q_max
        self.tol = tol
        self.status = "Success"
        self.constants()
        self.c = c
        self.dmu_q = dmu_q
        self.dmu_q_factor = dmu_q_factor
        self.filename_crust = filename_crust
        self.build_CFL_EoS()



        self.N_low_dens = N_low_dens
        self.rho_min_low_dens = rho_min_low_dens
        self.rho_max_low_dens = rho_max_low_dens
        self.rho_max_high_dens = rho_max_high_dens
        self.d_rho = d_rho
        self.eos_name = eos_name
        self.mix_phase = mix_phase
        self.TOV = TOV
        self.TOV_limit = TOV_limit
        #The next if statement checks if a low density eos is given,
        #otherwise, create one based on the name given in eos_name
        if(eos_low_dens!=None):
            self.eos_low_dens=eos_low_dens

        elif(self.eos_name == 'APR'):
            self.eos_low_dens = APR03.apr03_EoS(self.N_low_dens
                                 ,rho_min = self.rho_min_low_dens
                                 ,rho_max = self.rho_max_low_dens
                                 ,d_rho = self.d_rho)
        elif(self.eos_name == "RMF"):
            self.eos_low_dens = RMF.EOS_set(N=self.N_low_dens,rho_min=self.rho_min_low_dens,rho_max=self.rho_max_low_dens,RMF_filename=RMF_filename)
        else:
            self.status="Fail, low density eos not supported"
            print("EoS not supported")

        #Create interpolation tables with the low density eos. To do so we have to remove possible nan values
        self.e_NM_vec = self.eos_low_dens.e_vec
        self.P_NM_vec = self.eos_low_dens.P_vec
        self.rho_NM_vec = self.eos_low_dens.rho_vec
        self.mu_e_NM_vec = self.eos_low_dens.mu_e_vec
        self.mu_q_NM_vec = self.eos_low_dens.mu_n_vec/3
        self.xp_NM_vec = self.eos_low_dens.xp_vec

        if(self.eos_name=="RMF"):
            self.S_NM_vec = self.eos_low_dens.fields_vec[:,0]
            self.V_NM_vec = self.eos_low_dens.fields_vec[:,1]
            self.D_NM_vec = self.eos_low_dens.fields_vec[:,2]
            self.B_NM_vec = self.eos_low_dens.fields_vec[:,3]

        mask_P = self.remove_nan_mask(self.P_NM_vec,self.mu_q_NM_vec)
        mask_rho = self.remove_nan_mask(self.rho_NM_vec,self.mu_q_NM_vec)
        mask_xp = self.remove_nan_mask(self.xp_NM_vec,self.mu_q_NM_vec)
        mask_e = self.remove_nan_mask(self.e_NM_vec,self.mu_q_NM_vec)
        mask_mu_e = self.remove_nan_mask(self.mu_e_NM_vec,self.mu_q_NM_vec)
        mask_S = self.remove_nan_mask(self.S_NM_vec,self.mu_q_NM_vec)
        mask_V = self.remove_nan_mask(self.V_NM_vec,self.mu_q_NM_vec)
        mask_B = self.remove_nan_mask(self.B_NM_vec,self.mu_q_NM_vec)
        mask_D = self.remove_nan_mask(self.D_NM_vec,self.mu_q_NM_vec)


        order = "linear"
        self.e_of_mu_q_low_dens = interp1d(self.mu_q_NM_vec[mask_e],self.e_NM_vec[mask_e],kind=order,fill_value="extrapolate")
        self.P_of_mu_q_low_dens = interp1d(self.mu_q_NM_vec[mask_P],self.P_NM_vec[mask_P],kind=order,fill_value="extrapolate")

        self.rho_of_mu_q_low_dens = interp1d(self.mu_q_NM_vec[mask_rho],self.rho_NM_vec[mask_rho],kind=order,fill_value="extrapolate")
        self.mu_e_of_mu_q_low_dens = interp1d(self.mu_q_NM_vec[mask_mu_e],self.mu_e_NM_vec[mask_mu_e],kind=order,fill_value="extrapolate")
        self.xp_of_mu_q_low_dens = interp1d(self.mu_q_NM_vec[mask_xp],self.xp_NM_vec[mask_xp],kind=order,fill_value="extrapolate")
        if(self.eos_name == "RMF"):
            self.S_of_mu_q_low_dens = interp1d(self.mu_q_NM_vec[mask_S],self.S_NM_vec[mask_S],kind=order,fill_value="extrapolate")
            self.V_of_mu_q_low_dens = interp1d(self.mu_q_NM_vec[mask_V],self.S_NM_vec[mask_V],kind=order,fill_value="extrapolate")
            self.B_of_mu_q_low_dens = interp1d(self.mu_q_NM_vec[mask_B],self.S_NM_vec[mask_B],kind=order,fill_value="extrapolate")
            self.D_of_mu_q_low_dens = interp1d(self.mu_q_NM_vec[mask_D],self.S_NM_vec[mask_D],kind=order,fill_value="extrapolate")

        self.mu_q_of_rho_low_dens = interp1d(self.rho_NM_vec[mask_rho],self.mu_q_NM_vec[mask_rho],kind=order,fill_value="extrapolate")
        self.P_of_e_low_dens = interp1d(self.e_NM_vec[mask_e],self.P_NM_vec[mask_e],kind=order,fill_value="extrapolate")
        self.P_of_rho_low_dens = interp1d(self.rho_NM_vec[mask_P],self.P_NM_vec[mask_P],kind=order,fill_value="extrapolate")


        #If no errors have occured so far, we try to create a kaon phase
        if(self.status=="Success"):
            if(self.mix_phase==True):
                self.build_CFL_kaons_EoS()
            else:
                self.status="no mix"
            self.full_eos()
        #else:
            #q = 1
            #print("Warning, status: "+str(self.status))
        #If no errors occured, create mass radius realtions, given the flag TOV is true
        if(self.TOV==True):
            if(self.status!="Fail"):
                self.add_crust()
                if(sum(np.isnan(self.v2_w_crust_vec))>0 or min(self.v2_w_crust_vec)<0):
                    self.status="Fail"
            if(self.status!="Fail"):
                self.rho_of_P = interp1d(self.P_w_crust_vec,self.rho_w_crust_vec,kind="linear",fill_value="extrapolate")
                try:
                    R_vec,M_vec,Lambda_vec,P_c_vec = TOV_Rahul.tov_solve(self.e_w_crust_vec,self.P_w_crust_vec,self.v2_w_crust_vec,TOV_limit=self.TOV_limit)
                    self.R_vec = R_vec
                    self.M_vec = M_vec
                    self.Lambda_vec = Lambda_vec
                    self.P_c_vec = P_c_vec
                    self.rho_c_vec = self.rho_of_P(P_c_vec)
                    self.remove_unstable_end()
                except:
                    self.R_vec = np.zeros(100)
                    self.M_vec = np.zeros(100)
                    self.Lambda_vec = np.zeros(100)
                    self.P_c_vec = np.zeros(100)
                    self.rho_c_vec = np.zeros(100)
                    self.status="Fail"

            else:
                self.R_vec = np.zeros(100)
                self.M_vec = np.zeros(100)
                self.Lambda_vec = np.zeros(100)
                self.P_c_vec = np.zeros(100)
        return

    def remove_nan_mask(self,x,y):
        return (~np.isnan(x))*(~np.isnan(y))

    def constants(self):
        self.m_u = 3.75         #Mass of up quark in MeV
        self.m_d = 7.5          #Mass of down quark in MeV
        self.m_mu = 105.66      #Mass of muon in MeV
        self.m_e = 0.50998      #Mass of electron in MeV
        self.hc = 197.327       #hbar times speed of light in MeV

    def lepton_energy_density(self,mu,m_l):
        if(mu<m_l):
            return 0
        pF = np.sqrt(mu**2-m_l**2)
        return ((pF*mu**3)+(pF**3*mu)+m_l**4*np.log(mu/(pF+mu)))/(8*np.pi**2*self.hc**3)


    #Fermi momentum functional for quarks for a given quark fermi momentum pF and chemical potential mu_q
    def fermi_momenta_functional_CFL(self,pF,mu_q):
        return (np.sqrt(self.m_s**2+pF**2)
               +np.sqrt(self.m_u**2+pF**2)
               +np.sqrt(self.m_d**2+pF**2)
               -3*mu_q)

    #find zeros of fermi_momenta_functional_CFL for a given mu_q to determine quark fermi momentum
    def fermi_momenta_CFL(self,mu_q,pF_guess):
        return optimize.fsolve(self.fermi_momenta_functional_CFL,[pF_guess],args=(mu_q,))

    #Energy density of a free gas of quarks in the CFL phase
   # def energy_density_free_quark_CFL(self,pF,mu_q,m_q):
   #     return ((pF*mu_q**3)+(pF**3*mu_q)+m_q**4*np.log(mu_q/(pF+mu_q)))/(8*np.pi**2)

    def energy_density_free_quark_CFL(self,pF,m_q):
        if(m_q == 0):
            return pF**4/(4*np.pi**2)
        return (2*pF*(pF**2+m_q**2)**(3/2)
                -m_q**2*pF*np.sqrt(pF**2+m_q**2)
                +m_q**4*np.log(m_q/(pF+np.sqrt(pF**2+m_q**2))))/(8*np.pi**2)

    #Energy density with interactions in the CFL phase
    def energy_density_CFL(self,mu_q,pF):
        return (3*(1-self.c)*self.energy_density_free_quark_CFL(pF,self.m_u)
               +3*(1-self.c)*self.energy_density_free_quark_CFL(pF,self.m_d)
               +3*self.energy_density_free_quark_CFL(pF,self.m_s)
               -3*self.c*self.energy_density_free_quark_CFL(pF,0)
               -(9*self.c*mu_q*pF**2/np.pi**2)*(2-mu_q/np.sqrt(mu_q**2+self.m_s**2/3))*(mu_q-pF)
               +3*mu_q**2*self.Delta**2/np.pi**2+self.B**4)/self.hc**3

    #Baryon density in the CFL phase
    def density_CFL(self,mu_q,pF):
        return ((1-self.c)*3*pF**3/np.pi**2
                -9*self.c*pF**2/np.pi**2*(2-mu_q/np.sqrt(mu_q**2+self.m_s**2/3))*(mu_q-pF)
                +6*mu_q*self.Delta**2/np.pi**2)/self.hc**3

    #Pressure of free quarks in the CFL phase
    def pressure_free_quark_CFL(self,mu_q,pF,m_q):
        return mu_q*pF**3/(3*np.pi**2)-self.energy_density_free_quark_CFL(pF,m_q)


    #Pressure with interactions in CFL phase
    def pressure_CFL(self,mu_q,pF):
        return (3*(1-self.c)*self.pressure_free_quark_CFL(mu_q,pF,self.m_u)
               +3*(1-self.c)*self.pressure_free_quark_CFL(mu_q,pF,self.m_d)
               +3*self.pressure_free_quark_CFL(mu_q,pF,self.m_s)
               -3*self.c*self.pressure_free_quark_CFL(mu_q,pF,0)
               +3*self.Delta**2*mu_q**2/np.pi**2-self.B**4)/self.hc**3

    #Build the entire CFL table (Note, this is without the kaon condensate)
    def build_CFL_EoS(self):
        self.mu_q_CFL_vec = np.linspace(self.mu_q_min,self.mu_q_max,self.N_CFL)
        self.pF_CFL_vec = np.zeros(self.N_CFL)
        self.e_CFL_vec = np.zeros(self.N_CFL)
        self.rho_CFL_vec = np.zeros(self.N_CFL)
        self.P_CFL_test_vec = np.zeros(self.N_CFL)
        pF_guess = self.mu_q_CFL_vec[0]
        for i in range(self.N_CFL):
            mu_q = self.mu_q_CFL_vec[i]
            self.pF_CFL_vec[i] = self.fermi_momenta_CFL(mu_q,pF_guess)
            pF = self.pF_CFL_vec[i]
            pF_guess = pF
            self.e_CFL_vec[i] = self.energy_density_CFL(mu_q,pF)
            self.rho_CFL_vec[i] = self.density_CFL(mu_q,pF)

            #Sanity check for pressure. Should be equal to P_CFL_vec
            self.P_CFL_test_vec[i] = self.pressure_CFL(self.mu_q_CFL_vec[i],self.pF_CFL_vec[i])

        order = "linear"
        self.P_CFL_vec = self.mu_q_CFL_vec*self.rho_CFL_vec-self.e_CFL_vec
        self.P_of_mu_q_CFL = interp1d(self.mu_q_CFL_vec,self.P_CFL_vec,kind=order,fill_value="extrapolate")
        self.e_of_mu_q_CFL = interp1d(self.mu_q_CFL_vec,self.e_CFL_vec,kind=order,fill_value="extrapolate")
        self.rho_of_mu_q_CFL = interp1d(self.mu_q_CFL_vec,self.rho_CFL_vec,kind=order,fill_value="extrapolate")
        self.P_of_rho_CFL = interp1d(self.rho_CFL_vec,self.P_CFL_vec,kind=order,fill_value="extrapolate")
        self.e_of_rho_CFL = interp1d(self.rho_CFL_vec,self.e_CFL_vec,kind=order,fill_value="extrapolate")
        self.pF_of_rho_CFL = interp1d(self.rho_CFL_vec,self.pF_CFL_vec,kind=order,fill_value="extrapolate")
        self.P_of_e_CFL = interp1d(self.e_CFL_vec,self.P_CFL_vec,kind=order,fill_value="extrapolate")
        self.pF_of_mu_q_CFL = interp1d(self.mu_q_CFL_vec,self.pF_CFL_vec,kind=order,fill_value="extrapolate")

        return


    #
    # What comes next is regarding a kaon condensate coexisting during the transition
    #

    #Equation 34 in xeos note
    def f_pi_squared(self,mu_q):
        return (21-8*np.log(2))/(36*np.pi**2)*mu_q**2

    #Effective mass of a kaon squared
    def kaon_mass_squared(self,mu_q):
        return 3*self.Delta**2/(self.f_pi_squared(mu_q)*np.pi**2)*self.m_d*(self.m_u+self.m_s)

    #Adding a kaon condensate to the CFL EoS
    def pressure_CFL_kaons(self,mu_q,pF,mu_e):
        if(self.kaon_mass_squared(mu_q)<mu_e**2 and mu_e>0):
            if(mu_e>self.m_e):
                pF_e = np.sqrt(mu_e**2-self.m_e**2)
            else:
                pF_e = 0
            if(mu_e>self.m_mu):
                pF_mu = np.sqrt(mu_e**2-self.m_mu**2)
            else:
                pF_mu = 0

            rho_e = pF_e**3/(3*np.pi**2*self.hc**3)
            rho_mu = pF_mu**3/(3*np.pi**2*self.hc**3)

            return (self.pressure_CFL(mu_q,pF)+self.f_pi_squared(mu_q)*mu_e**2*(1-self.kaon_mass_squared(mu_q)/mu_e**2)**2/(2*self.hc**3)
                    +mu_e*rho_e+mu_e*rho_mu
                    -self.lepton_energy_density(mu_e,self.m_e)
                    -self.lepton_energy_density(mu_e,self.m_mu))
        return self.pressure_CFL(mu_q,pF)

    #Total energy density of CFL when kaons are included
    def energy_density_CFL_kaons(self,mu_q,mu_e):
        if(self.kaon_mass_squared(mu_q)>mu_e**2 or mu_e<0):
            return 0
        return (self.f_pi_squared(mu_q)*mu_e**2*(
                1+2*self.kaon_mass_squared(mu_q)/mu_e**2-3*self.kaon_mass_squared(mu_q)**2/mu_e**4)/(2*self.hc**3)
                +self.lepton_energy_density(mu_e,self.m_e)
                +self.lepton_energy_density(mu_e,self.m_mu))

    #Kaon number density
    def kaon_density_CFL(self,mu_q,mu_e):
        if(self.kaon_mass_squared(mu_q)>mu_e**2 or mu_e<0):
            return 0
        return self.f_pi_squared(mu_q)*mu_e*(1-self.kaon_mass_squared(mu_q)**2/mu_e**4)/(self.hc**3)

    #Functional used to find the electron chemical potential for the mixed phase
    def chemical_potential_electron_CFL_kaons_functional(self,mu_e_xp_rho,mu_q,pF):
        if(self.eos_name=="APR"):
            mu_e,xp,rho = mu_e_xp_rho
        elif(self.eos_name=="RMF"):
            mu_e,xp,rho,kFn,kFp,S,V,B,D=mu_e_xp_rho
            fields = [S,V,B,D]

        mu_n = 3*mu_q
        mu_p = mu_n-mu_e

        if(mu_e>self.m_e):
            kFe = np.sqrt(mu_e**2-self.m_e**2)
        else:
            kFe = 0

        if(mu_e>self.m_mu):
            kFmu = np.sqrt(mu_e**2-self.m_mu**2)
        else:
            kFmu = 0

        if(self.eos_name=="APR"):
            mu_n_calc,mu_p_calc = self.eos_low_dens.chemical_potential_functional(rho,xp,1)
            f1 = (self.eos_low_dens.total_pressure_of_mu_e(mu_n,mu_e,xp,rho,1)
                    -self.pressure_CFL_kaons(mu_q,pF,mu_e))
            f2 = mu_n-mu_n_calc
            f3 = mu_p-mu_p_calc
            return f1,f2,f3

        elif(self.eos_name=="RMF"):
            kFn_calc = self.eos_low_dens.get_kF(rho*(1-xp)*self.eos_low_dens.hbc3)
            kFp_calc = self.eos_low_dens.get_kF(rho*xp*self.eos_low_dens.hbc3)
            f6,f7,f8,f9 = self.eos_low_dens.mesons_kaons_functional(rho*self.eos_low_dens.hbc3,xp,kFn,kFp,fields)

            mu_p_calc,mu_n_calc = self.eos_low_dens.get_chem_pot_kaons(xp,kFn,kFp,fields)

            f1 = self.eos_low_dens.p_QHD_kaons(rho*self.eos_low_dens.hbc3,xp,kFe,kFmu,kFn,kFp,*fields)/self.eos_low_dens.hbc3-self.pressure_CFL_kaons(mu_q,pF,mu_e)

            f2 = mu_n-mu_n_calc
            f3 = mu_p-mu_p_calc
            f4 = kFn-kFn_calc
            f5 = kFp-kFp_calc
            return f1,f2,f3,f4,f5,f6,f7,f8,f9

    #Find the zeros of chemical_potential_electron_CFL_kaons_functional in order to
    #determine electron chemical potential, proton fraction, and total density
    def chemical_potential_electron_CFL_kaons(self,mu_q,pF,mu_e_xp_rho_guess,start):
        #print(mu_e_xp_rho_guess)
        mu_e_xp_rho,info,status,message=(
        optimize.fsolve(self.chemical_potential_electron_CFL_kaons_functional,mu_e_xp_rho_guess,args=(mu_q,pF),full_output= True))
        if(status!=1 and start==True):
            new_guess = [entry*1.1 for entry in mu_e_xp_rho_guess]
            mu_e_xp_rho,info,status,message=(
            optimize.fsolve(self.chemical_potential_electron_CFL_kaons_functional,new_guess,args=(mu_q,pF),full_output= True))
            if(status!=1):
                new_guess = [entry*0.9 for entry in mu_e_xp_rho_guess]
                mu_e_xp_rho,info,status,message=(
                optimize.fsolve(self.chemical_potential_electron_CFL_kaons_functional,new_guess,args=(mu_q,pF),full_output= True))



        return mu_e_xp_rho,status

    def kaon_phase_values(self,mu_q,pF_guess,mu_e,xp,S,V,B,D,kFn,kFp,start):
        pF = self.fermi_momenta_CFL(mu_q,pF_guess)[0]
        pF_guess = pF

        #Compute the neutron matter electron chemical potential
        mu_e_NM = self.mu_e_of_mu_q_low_dens(mu_q)
        xp_NM = self.xp_of_mu_q_low_dens(mu_q)
        rho_NM = self.rho_of_mu_q_low_dens(mu_q)
        if(self.eos_name=="RMF"):
            fields_NM = [self.S_of_mu_q_low_dens(mu_q),
                         self.V_of_mu_q_low_dens(mu_q),
                         self.B_of_mu_q_low_dens(mu_q),
                         self.D_of_mu_q_low_dens(mu_q)]
        else:
            fields_NM = None


        #Check if the "converge" flag is true. This flag is
        #set by the solver for that finds the electron chemical
        #potential in the mix phase converged.
        #This solver is used further down in the code
        if(start==False):
            #If the solver for the mix phase failed, that is
            #usually because we are not in the mix phase yet.
            #Therefore, set all initallial guesses for the
            #next time the solver is used to the nuclear model
            #values.
            xp_guess = xp_NM#self.xp_of_mu_q_low_dens(mu_q)
            rho_guess = rho_NM#self.rho_of_mu_q_low_dens(mu_q)
            kFn_guess = (rho_guess*3*np.pi**2)**(1/3)
            kFp_guess = (xp_guess*rho_guess*3*np.pi**2)**(1/3)
            mu_e_guess = mu_e_NM
            if(self.eos_name=="RMF"):
                [S_guess,V_guess,B_guess,D_guess]=fields_NM
        else:
            #If the mix phase solver converged, use the values in that
            #phase as guesses for the next iteration. These values are
            #marked with an "_c" to denote convergence.
            if(self.eos_name == "RMF"):
                kFn_guess = kFn
                kFp_guess = kFp
                S_guess = S
                B_guess = B
                V_guess = V
                D_guess = D

            mu_e_guess = mu_e
            xp_guess = xp
            rho_guess = rho_NM

        #If we use the APR nuclear EoS, only electron chemical potential
        #proton fraction and density (mu_e,xp,rho) are needed for the
        #mix phase solver
        if(self.eos_name == "APR"):
            mu_e_xp_rho_guess = [mu_e_guess,xp_guess,rho_guess]

        #If we use the RMF model, we need electron chemical potential,
        #density, neutron and proton Fermi momenta, the fields (mu_e,rho,kFn,kFp,S,V,B,D)
        # in the mix phase solver
        elif(self.eos_name == "RMF"):
            mu_e_xp_rho_guess = [mu_e_guess,xp_guess,rho_guess,kFn_guess,kFp_guess,S_guess,V_guess,B_guess,D_guess]

        #Run the solver that finds the electron chemical potential mu_e in the mix phase
        mu_e_xp_rho,status_chemical = self.chemical_potential_electron_CFL_kaons(mu_q,pF,mu_e_xp_rho_guess,start)

        if(mu_e_xp_rho[0]>0 and mu_e_xp_rho[0]<mu_e_NM and status_chemical==1 and abs(mu_e_NM-mu_e_xp_rho[0])/mu_e_NM<1 and start==False):
            start = True
            self.dmu_q = self.dmu_q/self.dmu_q_factor
        if(start==False):
            mu_e,xp,rho_NM,kFn,kFp,S,V,B,D = [0,0,0,0,0,0,0,0,0]
        if(start==True):
            if(self.eos_name == "APR"):
                mu_e,xp_NM,rho_NM = mu_e_xp_rho
            elif(self.eos_name == "RMF"):
                mu_e,xp_NM,rho_NM,kFn,kFp,S,V,B,D = mu_e_xp_rho
        P_CFL_kaons=self.pressure_CFL_kaons(mu_q,pF,mu_e)

        e_CFL_kaons = (self.energy_density_CFL(mu_q,pF)+self.energy_density_CFL_kaons(mu_q,mu_e))

        if(mu_e>self.m_e):
            pF_e = np.sqrt(mu_e**2-self.m_e**2)
        else:
            pF_e = 0

        if(mu_e>self.m_mu):
            pF_mu = np.sqrt(mu_e**2-self.m_mu**2)
        else:
            pF_mu = 0
        rho_e = pF_e**3/(3*np.pi**2*self.hc**3)
        rho_mu = pF_mu**3/(3*np.pi**2*self.hc**3)

        if(self.eos_name == "APR"):
            e_NM = (self.eos_low_dens.e_functional(rho_NM,xp)[1]
                    +self.eos_low_dens.lepton_energy_density(mu_e,self.m_e)
                    +self.eos_low_dens.lepton_energy_density(mu_e,self.m_mu))
        elif(self.eos_name == "RMF"):
            fields_NM = [S,V,B,D]
            e_NM = self.eos_low_dens.e_QHD(rho_NM*self.eos_low_dens.hbc3,xp_NM,pF_e,*fields_NM)/self.eos_low_dens.hbc3
        if(mu_e==0):
            if(start==False):
                chi = 0
            else:
                chi = 1
        else:
            chi = (xp_NM*rho_NM-rho_e-rho_mu)/(xp_NM*rho_NM+self.kaon_density_CFL(mu_q,mu_e))

        rho_CFL = self.density_CFL(mu_q,pF)+rho_e+rho_mu
        rho_CFL_kaons = rho_NM*(1-chi)+chi*rho_CFL
        e_CFL_kaons = (1-chi)*e_NM + chi*e_CFL_kaons
        rho_CFL_kaons = rho_NM*(1-chi)+chi*rho_CFL
        return mu_e,mu_e_NM,e_CFL_kaons,P_CFL_kaons,rho_CFL_kaons,pF,xp,rho_NM,B,D,V,S,kFn,kFp,start


    #Build EoS with mixed phase of CFL and kaons
    def build_CFL_kaons_EoS(self):
        #Initialize a bunch of empty vectors

        self.P_CFL_kaons_vec = []
        self.mu_q_CFL_kaons_vec = []
        self.mu_e_CFL_kaons_vec = []
        self.e_CFL_kaons_vec = []
        self.rho_CFL_kaons_vec = []
        self.pF_CFL_kaons_vec = []
        self.S_CFL_kaons_vec =[]
        self.V_CFL_kaons_vec =[]
        self.B_CFL_kaons_vec = []
        self.D_CFL_kaons_vec = []
        self.kFn_CFL_kaons_vec =[]
        self.kFp_CFL_kaons_vec = []
        self.xp_CFL_kaons_vec = []
        self.rho_NM_CFL_kaons_vec = []



        #for i in range(self.N_kaons):
        mu_q = self.mu_q_min-self.dmu_q
        #Check if the low density phase has lower pressure than the CFL phase
        #at the saturation density. If so, there is no CFL phase, and we return the
        #'no CFL' status

        if(self.P_of_rho_CFL(0.16)>self.P_of_rho_low_dens(0.16)):
            self.status="no CFL"
            return

        mu_q = self.mu_q_min-self.dmu_q
        pF_guess = self.pF_of_mu_q_CFL(mu_q)
        start = False
        status = 0
        self.mix_index_start = 0
        [mu_e,xp,S,V,B,D,kFn,kFp] = [0,0,0,0,0,0,0,0]
        while (start==False or (start==True and mu_e>0)):
            mu_q+=self.dmu_q
            mu_e,mu_e_NM,e_CFL_kaons,P_CFL_kaons,rho_CFL_kaons,pF,xp,rho_NM,B,D,V,S,kFn,kFp,start= self.kaon_phase_values(mu_q,pF_guess,mu_e,xp,S,V,B,D,kFn,kFp,start)
            if(start==False and self.rho_of_mu_q_low_dens(mu_q)>self.rho_max_low_dens):
                self.status = "no CFL"
                return
            if(start == True and rho_CFL_kaons<0.16):
                self.status="no CFL"
                return
            if(start==True and rho_CFL_kaons>self.rho_max_high_dens):
                break

        mu_q_current = mu_q
        abort = 0
        while(mu_e<mu_e_NM):
            mu_q-=self.dmu_q
            mu_e_prev = mu_e
            self.mu_q_CFL_kaons_vec.insert(0,mu_q)
            mu_e,mu_e_NM,e_CFL_kaons,P_CFL_kaons,rho_CFL_kaons,pF,xp,rho_NM,B,D,V,S,kFn,kFp,start = self.kaon_phase_values(mu_q,pF_guess,mu_e,xp,S,V,B,D,kFn,kFp,start)
            self.mu_e_CFL_kaons_vec.insert(0,mu_e)
            self.e_CFL_kaons_vec.insert(0,e_CFL_kaons)
            self.P_CFL_kaons_vec.insert(0,P_CFL_kaons)
            self.rho_CFL_kaons_vec.insert(0,rho_CFL_kaons)
            self.pF_CFL_kaons_vec.insert(0,pF)
            self.xp_CFL_kaons_vec.insert(0,xp)
            self.rho_NM_CFL_kaons_vec.insert(0,rho_NM)
            self.B_CFL_kaons_vec.insert(0,B)
            self.D_CFL_kaons_vec.insert(0,D)
            self.V_CFL_kaons_vec.insert(0,V)
            self.S_CFL_kaons_vec.insert(0,S)
            self.kFn_CFL_kaons_vec.insert(0,kFn)
            self.kFp_CFL_kaons_vec.insert(0,kFp)
            if(mu_e_prev>mu_e):
                abort+=1
            if(abort == 3):
                self.status = "no CFL"
                return

        mu_q = mu_q_current-self.dmu_q
        while(self.rho_of_mu_q_CFL(mu_q)<self.rho_max_high_dens):
            mu_q+=self.dmu_q
            self.mu_q_CFL_kaons_vec.append(mu_q)
            self.mu_e_CFL_kaons_vec.append(0)
            self.e_CFL_kaons_vec.append(self.e_of_mu_q_CFL(mu_q))
            self.P_CFL_kaons_vec.append(self.P_of_mu_q_CFL(mu_q))
            self.pF_CFL_kaons_vec.append(self.pF_of_mu_q_CFL(mu_q))
            self.rho_CFL_kaons_vec.append(self.rho_of_mu_q_CFL(mu_q))
            self.xp_CFL_kaons_vec.append(0)
            self.rho_NM_CFL_kaons_vec.append(0)
            self.B_CFL_kaons_vec.append(0)
            self.D_CFL_kaons_vec.append(0)
            self.V_CFL_kaons_vec.append(0)
            self.S_CFL_kaons_vec.append(0)
            self.kFn_CFL_kaons_vec.append(0)
            self.kFp_CFL_kaons_vec.append(0)



        #Code between horisontal dashed line walks backwards to add more points to CFL kaon phase
        #in case convergence did not go well initially
        #--------------


        #Create interpolation tables for the kaon phase from infinite quark chemical potential
        #and up until the transition. Extrapolate to make it easier to find transition point.
        order = "linear"


        self.mu_e_of_mu_q_kaons = (interp1d(self.mu_q_CFL_kaons_vec,
                                            self.mu_e_CFL_kaons_vec,
                                            kind=order,fill_value="extrapolate"))


        self.e_of_mu_q_kaons = (interp1d(self.mu_q_CFL_kaons_vec,
                                        self.e_CFL_kaons_vec,
                                        kind=order,fill_value="extrapolate"))


        self.P_of_mu_q_kaons = (interp1d(self.mu_q_CFL_kaons_vec,
                                        self.P_CFL_kaons_vec,
                                        kind=order,fill_value="extrapolate"))


        self.rho_of_mu_q_kaons = (interp1d(self.mu_q_CFL_kaons_vec,
                                        self.rho_CFL_kaons_vec,
                                        kind=order,fill_value="extrapolate"))

        self.P_of_e_kaons = (interp1d(self.e_CFL_kaons_vec,
                                        self.P_CFL_kaons_vec,
                                        kind=order,fill_value="extrapolate"))

        self.P_of_rho_kaons = (interp1d(self.rho_CFL_kaons_vec,
                                        self.P_CFL_kaons_vec,
                                        kind=order,fill_value="extrapolate"))


        self.mu_e_CFL_kaons_vec = self.mu_e_of_mu_q_kaons(self.mu_q_CFL_kaons_vec)
        self.e_CFL_kaons_vec = self.e_of_mu_q_kaons(self.mu_q_CFL_kaons_vec)
        self.P_CFL_kaons_vec = self.P_of_mu_q_kaons(self.mu_q_CFL_kaons_vec)
        self.rho_CFL_kaons_vec = self.rho_of_mu_q_kaons(self.mu_q_CFL_kaons_vec)

        return
    #Function that returns zero when pressure in CFL equal pressure in neutron matter
    def find_transition_mu_q_to_CFL_functional(self,mu_q):
        return (self.P_of_mu_q_CFL(mu_q)-self.P_of_mu_q_low_dens(mu_q))

    #function that finds transition form neutron matter to CFL
    def find_transition_mu_q_to_CFL(self,mu_q_guess):
        try:
            mu_q_intersect,info,status,message = optimize.fsolve(self.find_transition_mu_q_to_CFL_functional,mu_q_guess,full_output =
                    True)
            if(status!=1):
                self.status = "no CFL"
                #self.status="no mix"
        except:
            self.status = "no CFL"
            #self.status = "no mix"
            return 0

        return mu_q_intersect[0]

    def find_transition_mu_q_to_mixphase_functional(self,mu_q):
        #return self.P_of_e_kaons(mu_q)-self.P_of_e_low_dens(mu_q)
        return (self.mu_e_of_mu_q_kaons(mu_q)-self.mu_e_of_mu_q_low_dens(mu_q))
    #Finds the transition density from neutron matter to mixed phase by solving for zeros of find_transition_mu_q_to_mixphase_functional
    def find_transition_mu_q_to_mixphase(self,mu_q_guess):
        mu_q_guess = np.array(mu_q_guess)
        guess_vec = [mu_q_guess,1.02*mu_q_guess,0.98*mu_q_guess,1.05*mu_q_guess,0.95*mu_q_guess]
        for i in range(len(guess_vec)):
            mu_q_guess_ = guess_vec[i]
            try:
                mu_q_intersect,info,status,message = optimize.fsolve(self.find_transition_mu_q_to_mixphase_functional,mu_q_guess_,full_output=True)
            except:
                if(i==2):
                    self.status = "no CFL"
                    return 0
                continue
            if(status==1):
                break
        if(status!=1):
            self.status = "no CFL"
            return 0

        return mu_q_intersect[0]


    #Compute full equation of state with Apr03 or RMF at low density, kaon with CFL mixed phase, and pure CFL
    def full_eos(self):
        if(self.status == "Fail" or self.status == "no CFL"):
            self.P_vec = self.eos_low_dens.P_vec
            self.e_vec = self.eos_low_dens.e_vec
            self.rho_vec = self.eos_low_dens.rho_vec
            self.mu_q_vec = self.eos_low_dens.mu_n_vec/3
            self.mu_e_vec = self.eos_low_dens.mu_e_vec
            self.v2_vec = np.gradient(self.P_vec,self.e_vec,edge_order=2)
            #print("Warning in computing NM/kaon transition. Status: "+str(self.status))
            return

        if(len(self.mu_q_CFL_kaons_vec)>=2):
            self.mu_q_vec = np.concatenate((self.mu_q_NM_vec[self.mu_q_NM_vec<self.mu_q_CFL_kaons_vec[0]],self.mu_q_CFL_kaons_vec))
        else:
            self.status = "no CFL"
            self.P_vec = self.eos_low_dens.P_vec
            self.e_vec = self.eos_low_dens.e_vec
            self.rho_vec = self.eos_low_dens.rho_vec
            self.mu_q_vec = self.eos_low_dens.mu_n_vec/3
            self.mu_e_vec = self.eos_low_dens.mu_e_vec
            self.v2_vec = np.gradient(self.P_vec,self.e_vec,edge_order=2)
            return

        P_low_dens_vec = self.P_of_mu_q_low_dens(self.mu_q_vec)
        e_low_dens_vec = self.e_of_mu_q_low_dens(self.mu_q_vec)
        rho_low_dens_vec = self.rho_of_mu_q_low_dens(self.mu_q_vec)
        mu_e_low_dens_vec = self.mu_e_of_mu_q_low_dens(self.mu_q_vec)
        if(self.status!="no mix"):
            P_kaons_vec = self.P_of_mu_q_kaons(self.mu_q_vec)
            e_kaons_vec = self.e_of_mu_q_kaons(self.mu_q_vec)
            rho_kaons_vec = self.rho_of_mu_q_kaons(self.mu_q_vec)
            mu_e_kaons_vec = self.mu_e_of_mu_q_kaons(self.mu_q_vec)
        else:
            P_kaons_vec = self.P_of_mu_q_CFL(self.mu_q_vec)
            e_kaons_vec = self.e_of_mu_q_CFL(self.mu_q_vec)
            rho_kaons_vec = self.rho_of_mu_q_CFL(self.mu_q_vec)
            mu_e_kaons_vec = np.zeros(len(self.mu_q_vec))

        transition_index = abs(P_kaons_vec - P_low_dens_vec).argmin()

        if(transition_index<=1 or transition_index>=len(P_low_dens_vec)-2):
            self.status = "no CFL"
            self.P_vec = self.eos_low_dens.P_vec
            self.e_vec = self.eos_low_dens.e_vec
            self.rho_vec = self.eos_low_dens.rho_vec
            self.mu_q_vec = self.eos_low_dens.mu_n_vec/3
            self.mu_e_vec = self.eos_low_dens.mu_e_vec
            self.v2_vec = np.gradient(self.P_vec,self.e_vec,edge_order=2)
            return

        self.mu_q_trans =  optimize.fsolve(lambda mu_q:self.P_of_mu_q_low_dens(mu_q)-self.P_of_mu_q_kaons(mu_q),self.mu_q_vec[transition_index])[0]
        self.rho_trans = optimize.fsolve(lambda rho:self.P_of_rho_low_dens(rho)-self.P_of_rho_kaons(rho),rho_low_dens_vec[transition_index])[0]
        self.e_trans = optimize.fsolve(lambda e:self.P_of_e_low_dens(e)-self.P_of_e_kaons(e),e_low_dens_vec[transition_index])[0]
        self.P_trans = self.P_of_rho_low_dens(self.rho_trans)

        if(self.rho_trans>=rho_low_dens_vec[transition_index]):
            rho_low_dens_vec=rho_low_dens_vec[:transition_index+1]
            rho_low_dens_vec[-1] = self.rho_trans
            rho_kaons_vec = rho_kaons_vec[transition_index+1:]
        else:
            rho_low_dens_vec = rho_low_dens_vec[:transition_index]
            rho_low_dens_vec[-1] = self.rho_trans
            rho_kaons_vec = rho_kaons_vec[transition_index:]

        if(self.e_trans>=e_low_dens_vec[transition_index]):
            P_low_dens_vec = P_low_dens_vec[:transition_index+1]
            P_low_dens_vec[-1] = self.P_trans
            P_kaons_vec = P_kaons_vec[transition_index+1:]
            e_low_dens_vec = e_low_dens_vec[:transition_index+1]
            e_low_dens_vec[-1] = self.e_trans
            e_kaons_vec = e_kaons_vec[transition_index+1:]

        else:
            P_low_dens_vec = P_low_dens_vec[:transition_index]
            P_low_dens_vec[-1] = self.P_trans
            P_kaons_vec = P_kaons_vec[transition_index:]
            e_low_dens_vec = e_low_dens_vec[:transition_index]
            e_low_dens_vec[-1] = self.e_trans
            e_kaons_vec = e_kaons_vec[transition_index:]


        v2_1_vec = np.gradient(P_low_dens_vec,e_low_dens_vec,edge_order=2)
        v2_2_vec = np.gradient(P_kaons_vec,e_kaons_vec,edge_order=2)
        self.P_vec = np.concatenate((P_low_dens_vec,P_kaons_vec))
        self.rho_vec = np.concatenate((rho_low_dens_vec,rho_kaons_vec))
        self.mu_e_vec =  np.concatenate((mu_e_low_dens_vec,mu_e_kaons_vec))
        self.e_vec = np.concatenate((e_low_dens_vec,e_kaons_vec))
        self.v2_vec = np.concatenate((v2_1_vec,v2_2_vec))
        if(~np.all(self.P_vec[1:] >= self.P_vec[:-1]) or ~np.all(self.e_vec[1:]>= self.e_vec[:-1]) or np.any(self.v2_vec)<0 or np.any(np.isnan(self.v2_vec))):
            self.status = "Fail"
            self.P_vec = self.eos_low_dens.P_vec
            self.e_vec = self.eos_low_dens.e_vec
            self.rho_vec = self.eos_low_dens.rho_vec
            self.mu_q_vec = self.eos_low_dens.mu_n_vec/3
            self.mu_e_vec = self.eos_low_dens.mu_e_vec
            self.v2_vec = np.gradient(self.P_vec,self.e_vec,edge_order=2)
            return

        return




    def add_crust(self):
        P_conv = 8.96057e-7 #Convert between MeV/fm^3 to M_sol/km^3
        e_crust = []
        P_crust = []
        rho_crust = []
        o = 0
        for line in reversed(list(open(self.filename_crust,'r'))):
            e,P,rho = line.split()[:3]
            e = np.float64(e)/P_conv
            P = np.float64(P)/P_conv
            rho = np.float64(rho)
            e_crust.append(e)
            P_crust.append(P)
            rho_crust.append(rho)

        e_crust = np.array(e_crust)
        P_crust = np.array(P_crust)
        rho_crust = np.array(rho_crust)

        self.P_of_rho_crust = interp1d(rho_crust,P_crust,kind="linear",fill_value="extrapolate")
        self.P_of_rho = interp1d(self.rho_vec,self.P_vec,kind="linear",fill_value="extrapolate")
        rho_crust_trans = optimize.fsolve(lambda rho:self.P_of_rho_crust(rho)-self.P_of_rho(rho),0.08)[0]
        self.P_crust_vec = P_crust[rho_crust<=rho_crust_trans]
        self.e_crust_vec = e_crust[rho_crust<=rho_crust_trans]
        self.rho_crust_vec = rho_crust[rho_crust<=rho_crust_trans]

        self.v2_crust = np.gradient(np.array(self.P_crust_vec),np.array(self.e_crust_vec))
        self.P_w_crust_vec = np.concatenate((np.array(self.P_crust_vec),self.P_vec[self.P_vec>max(self.P_crust_vec)]))[1:]
        self.e_w_crust_vec = np.concatenate((np.array(self.e_crust_vec),self.e_vec[self.P_vec>max(self.P_crust_vec)]))[1:]
        self.rho_w_crust_vec = np.concatenate((np.array(self.rho_crust_vec),self.rho_vec[self.P_vec>max(self.P_crust_vec)]))[1:]
        self.v2_w_crust_vec = np.concatenate((self.v2_crust,self.v2_vec[self.P_vec>max(self.P_crust_vec)]))[1:]



    def remove_unstable_end(self):
        M = 0
        NN = len(self.M_vec)
        for i in range(NN):
            if(self.M_vec[NN-i-1]<=M):
                self.M_vec = self.M_vec[:NN-i+1]
                self.R_vec = self.R_vec[:NN-i+1]
                self.Lambda_vec = self.Lambda_vec[:NN-i+1]
                self.P_c_vec = self.P_c_vec[:NN-i+1]
                return
            M = self.M_vec[NN-i-1]
        return


