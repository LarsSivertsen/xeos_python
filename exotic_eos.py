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
                     ,N = 100                   #Number of points used to compute CFL phase
                     ,N_kaons = 100             #Number of points used to compute mixed kaon and CFL phase
                     ,N_low_dens = 400          #Number of points used in low density phase of the EoS                         #Bag constant
                     ,mu_q_min = 312.           #Minimum quark chemical potential we start computing CFL from [MeV]
                     ,mu_q_max = 700            #Maximum quark chemical potential we compute the CFL phase to [MeV]
                     ,rho_min_low_dens = 0.1             #Lowest number density we compute for the low density phaseAPR03 [fm^-3]
                     ,rho_max_low_dens = 3               #Max density we compute low density phase [fm^-3]
                     ,d_rho = 0.0001            #When computing derivatives in APR, we use this step size in rho
                     ,tol = 1e-6                #Absolute tolerance used in root solvers finding electron density and proton fraction
                     ,c = 0.                    #Phenomenological parameter for quark interactions. Usually set to 0.3
                     ,eos_name = "APR"          #Name of nuclear EoS (supports at the moment APR and RMF)

                     ,eos_low_dens = None       #If we already have computed the low density EoS, we can insert it here, so we don't have to
                                                #compute it again
                     ,RMF_filename = "FSUGoldgarnet.inp" #Filename for the parameters used in the RMF model
                     ,TOV=False                 #If set to True, compute MR curves in addition to EoS
                     ,mix_phase=True            #If set to False, there is no Kaon phase
                     ,filename_crust = "nveos.in"
                 ):

        self.N = N
        self.B = B
        self.Delta = Delta
        self.m_s = m_s
        self.mu_q_min = mu_q_min
        self.mu_q_max = mu_q_max
        self.tol = tol
        self.status = "Success"
        self.constants()
        self.c = c
        self.filename_crust = filename_crust
        self.build_CFL_EoS()


        self.N_kaons = N_kaons
        self.N_low_dens = N_low_dens
        self.rho_min_low_dens = rho_min_low_dens
        self.rho_max_low_dens = rho_max_low_dens
        self.d_rho = d_rho
        self.eos_name = eos_name
        self.mix_phase = mix_phase
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
            self.eos_low_dens = RMF.EOS_set(N=self.N_low_dens,rho_max=self.rho_max_low_dens,RMF_filename=RMF_filename)
        else:
            self.status="Fail, low density eos not supported"
            print("EoS not supported")

        if(self.status=="Success"):
            if(self.mix_phase==True):
                self.build_CFL_with_kaons_EoS()
            self.full_eos()

        self.add_crust()
        self.v2_vec = np.gradient(self.P_vec,self.e_vec,edge_order=2)
        self.v2_w_crust_vec = np.gradient(self.P_w_crust_vec,self.e_w_crust_vec,edge_order=2)
        if(TOV==True):
            R_vec,M_vec,Lambda_vec,P_c_vec = TOV_Rahul.tov_solve(self.e_w_crust_vec,self.P_w_crust_vec,self.v2_w_crust_vec)
            self.R_vec = R_vec
            self.M_vec = M_vec
            self.Lambda_vec = Lambda_vec
            self.P_c_vec = P_c_vec

        return

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
        self.mu_q_CFL_vec = np.linspace(self.mu_q_min,self.mu_q_max,self.N)
        self.pF_CFL_vec = np.zeros(self.N)
        self.e_CFL_vec = np.zeros(self.N)
        self.rho_CFL_vec = np.zeros(self.N)
        self.P_CFL_test_vec = np.zeros(self.N)
        pF_guess = self.mu_q_CFL_vec[0]
        for i in range(self.N):
            mu_q = self.mu_q_CFL_vec[i]
            self.pF_CFL_vec[i] = self.fermi_momenta_CFL(mu_q,pF_guess)
            pF = self.pF_CFL_vec[i]
            pF_guess = pF
            self.e_CFL_vec[i] = self.energy_density_CFL(mu_q,pF)
            self.rho_CFL_vec[i] = self.density_CFL(mu_q,pF)

            #Sanity check for pressure. Should be equal to P_CFL_vec
            self.P_CFL_test_vec[i] = self.pressure_CFL(self.mu_q_CFL_vec[i],self.pF_CFL_vec[i])

        self.P_CFL_vec = self.mu_q_CFL_vec*self.rho_CFL_vec-self.e_CFL_vec
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
    def pressure_CFL_with_kaons(self,mu_q,pF,mu_e):
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
    def energy_density_CFL_with_kaons(self,mu_q,mu_e):
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


    def chemical_potential_electron_CFL_with_kaons_functional(self,mu_e_xp_rho,mu_q,pF,fields_guess=[300,200,-10,-10]):
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
                    -self.pressure_CFL_with_kaons(mu_q,pF,mu_e))
            f2 = mu_n-mu_n_calc
            f3 = mu_p-mu_p_calc
            return f1,f2,f3

        elif(self.eos_name=="RMF"):
            S_guess = np.interp(rho,self.eos_low_dens.rho_vec,self.eos_low_dens.fields_vec[:,0])
            V_guess = np.interp(rho,self.eos_low_dens.rho_vec,self.eos_low_dens.fields_vec[:,1])
            B_guess = np.interp(rho,self.eos_low_dens.rho_vec,self.eos_low_dens.fields_vec[:,2])
            D_guess = np.interp(rho,self.eos_low_dens.rho_vec,self.eos_low_dens.fields_vec[:,3])
            fields_guess = np.array([S_guess,V_guess,B_guess,D_guess])
            fields_guess[np.isnan(np.array(fields_guess))]=0
            fields_guess = list(fields_guess)

            kFn_calc = self.eos_low_dens.get_kF(rho*(1-xp)*self.eos_low_dens.hbc3)
            kFp_calc = self.eos_low_dens.get_kF(rho*xp*self.eos_low_dens.hbc3)
            f6,f7,f8,f9 = self.eos_low_dens.mesons_kaons_functional(rho*self.eos_low_dens.hbc3,xp,kFn,kFp,fields)

            mu_p_calc,mu_n_calc = self.eos_low_dens.get_chem_pot_kaons(xp,kFn,kFp,fields)

            f1 = self.eos_low_dens.p_QHD_kaons(rho*self.eos_low_dens.hbc3,xp,kFe,kFmu,kFn,kFp,*fields)/self.eos_low_dens.hbc3-self.pressure_CFL_with_kaons(mu_q,pF,mu_e)

            f2 = mu_n-mu_n_calc
            f3 = mu_p-mu_p_calc
            f4 = kFn-kFn_calc
            f5 = kFp-kFp_calc
            return f1,f2,f3,f4,f5,f6,f7,f8,f9

    #Find the zeros of chemical_potential_electron_CFL_kaons_functional in order to
    #determine electron chemical potential, proton fraction, and total density
    def chemical_potential_electron_CFL_with_kaons(self,mu_q,pF,mu_e_xp_rho_guess,fields_guess = [300,200,-10,-10]):
        mu_e_xp_rho,info,status,message=(
        optimize.fsolve(self.chemical_potential_electron_CFL_with_kaons_functional,mu_e_xp_rho_guess,args=(mu_q,pF,fields_guess),full_output= True))
        return mu_e_xp_rho,status

    #Build EoS with mixed phase of CFL and kaons
    def build_CFL_with_kaons_EoS(self):
        self.mu_e_CFL_with_kaons_vec = np.zeros(self.N_kaons)
        self.mu_e_CFL_with_kaons_vec_test = np.zeros(self.N_kaons)
        self.P_CFL_with_kaons_vec = np.zeros(self.N_kaons)
        self.mu_q_CFL_with_kaons_vec = np.linspace(self.mu_q_min,self.mu_q_max,self.N_kaons)
        self.e_CFL_with_kaons_vec = np.zeros(self.N_kaons)
        self.rho_CFL_with_kaons_vec = np.zeros(self.N_kaons)
        status = 1
        start = False
        end = False
        pF_guess = 10
        self.mix_index_start = 0
        self.mix_index_end = 0
        fields_NM = [0,0,0,0]
        if(self.eos_name=="APR"):
            fields = None
            fields_guess = [0,0,0,0]
        for i in range(self.N_kaons):
            #Check if the low density phase has lower pressure than the CFL phase
            #at zero density. If so, there is no CFL phase, and we return the
            #'no CFL' status
            if(self.P_CFL_vec[0]>self.eos_low_dens.P_vec[0]):
                self.status=="no CFL"
                return

            #Use the quark chemical potential to compute quark Fermi momenta pF
            #
            mu_q = self.mu_q_CFL_with_kaons_vec[i]
            pF = self.fermi_momenta_CFL(mu_q,pF_guess)[0]
            pF_guess = pF

            mu_e_NM = np.interp(mu_q,self.eos_low_dens.mu_n_vec/3,self.eos_low_dens.mu_e_vec)

            if(start==False):
                xp_guess = np.interp(mu_q,self.eos_low_dens.mu_n_vec/3,self.eos_low_dens.xp_vec)
                rho_guess = np.interp(mu_q,self.eos_low_dens.mu_n_vec/3,self.eos_low_dens.rho_vec)
                kFn_guess = (rho_guess*3*np.pi**2)**(1/3)
                kFp_guess = (xp_guess*rho_guess*3*np.pi**2)**(1/3)
                mu_e_guess = mu_e_NM
                [S_guess,V_guess,B_guess,D_guess]=fields_NM

            else:
                if(self.eos_name == "RMF"):
                    kFn_guess = kFn
                    kFp_guess = kFp
                    S_guess = S
                    B_guess = B
                    V_guess = V
                    D_guess = D
                else:
                    S_guess = 0
                    V_guess = 0
                    B_guess = 0
                    D_guess = 0

                mu_e_guess = mu_e
                xp_guess = xp
                rho_guess = rho_NM

            fields_guess = [S_guess,V_guess,B_guess,D_guess]
            if(self.eos_name == "APR"):
                mu_e_xp_rho_guess = [mu_e_guess,xp_guess,rho_guess]
            elif(self.eos_name == "RMF"):
                mu_e_xp_rho_guess = [mu_e_guess,xp_guess,rho_guess,kFn_guess,kFp_guess,S_guess,V_guess,B_guess,D_guess]
            #print(fields_guess)
            if(end!=True):
                mu_e_xp_rho,status = self.chemical_potential_electron_CFL_with_kaons(mu_q,pF,mu_e_xp_rho_guess,fields_guess = fields_guess)
            #print(mu_e_xp_rho_guess,mu_e_xp_rho)

            if(status == 1 and mu_e_xp_rho[0]>0 and mu_e_xp_rho[0]<mu_e_NM):
                if(self.eos_name == "APR"):
                    mu_e,xp,rho_NM = mu_e_xp_rho
                elif(self.eos_name == "RMF"):
                    mu_e,xp,rho_NM,kFn,kFp,S,V,B,D = mu_e_xp_rho

                if(start==False):
                    self.mix_index_start = i
                    try:
                        j=0
                        while(i-1-j>=0):
                            self.mu_e_CFL_with_kaons_vec[i-1-j] = np.interp(self.mu_q_CFL_with_kaons_vec[i],self.eos_low_dens.mu_n_vec/3,self.eos_low_dens.mu_e_vec)
                            self.e_CFL_with_kaons_vec[i-1-j] = np.interp(self.mu_q_CFL_with_kaons_vec[i],self.eos_low_dens.mu_n_vec/3,self.eos_low_dens.e_vec)
                            self.P_CFL_with_kaons_vec[i-1-j] = np.interp(self.mu_q_CFL_with_kaons_vec[i],self.eos_low_dens.mu_n_vec/3,self.eos_low_dens.P_vec)
                            self.rho_CFL_with_kaons_vec[i-1-j] = np.interp(self.mu_q_CFL_with_kaons_vec[i],self.eos_low_dens.mu_n_vec/3,self.eos_low_dens.rho_vec)
                            j+=1
                    except:
                        pass
                start = True
            else:
                mu_e,xp,rho_NM,kFn,kFp,S,V,B,D = [0,0,0,0,0,0,0,0,0]
            #mu_e,xp,rho_NM,kFn,kFp,S,V,B,D = mu_e_xp_rho
            #print("rho:",rho_NM,"S:",S/self.eos_low_dens.gs2," V:",V/self.eos_low_dens.gv2," B:",B/self.eos_RMF.gp2," D:",D/self.eos_RMF.gd2)


            self.P_CFL_with_kaons_vec[i] = self.pressure_CFL_with_kaons(mu_q,pF,mu_e)
            self.mu_e_CFL_with_kaons_vec[i] = mu_e

            e_CFL_kaons = (self.energy_density_CFL(mu_q,pF)+self.energy_density_CFL_with_kaons(mu_q,mu_e))


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
                fields_NM = self.eos_low_dens.solve_mesons(rho_guess*self.eos_low_dens.hbc3,xp_guess,fields_guess)
                e_NM = self.eos_low_dens.e_QHD(rho_NM*self.eos_low_dens.hbc3,xp,pF_e,*fields_NM)/self.eos_low_dens.hbc3


            if(mu_e==0):
                if(start==False):
                    chi = 0
                else:
                    chi = 1
            else:
                chi = (xp*rho_NM-rho_e-rho_mu)/(xp*rho_NM+self.kaon_density_CFL(mu_q,mu_e))


            rho_CFL_with_kaons = self.density_CFL(mu_q,pF)+rho_e+rho_mu
            self.e_CFL_with_kaons_vec[i] = (1-chi)*e_NM + chi*e_CFL_kaons
            self.rho_CFL_with_kaons_vec[i] = rho_NM*(1-chi)+chi*rho_CFL_with_kaons

        try:
            self.mu_e_of_mu_q_kaons = (interp1d(self.mu_q_CFL_with_kaons_vec[self.mix_index_start:],
                                            self.mu_e_CFL_with_kaons_vec[self.mix_index_start:],
                                            kind='quadratic',fill_value="extrapolate"))
        except:
            self.mu_e_of_mu_q_kaons = (interp1d(self.mu_q_CFL_with_kaons_vec[self.mix_index_start:],
                                            self.mu_e_CFL_with_kaons_vec[self.mix_index_start:],
                                            kind='linear',fill_value="extrapolate"))

        try:
            self.e_of_mu_q_kaons = interp1d(self.mu_q_CFL_with_kaons_vec[self.mix_index_start:],
                                        self.e_CFL_with_kaons_vec[self.mix_index_start:],
                                        kind="quadratic",fill_value="extrapolate")
        except:
            self.e_of_mu_q_kaons = interp1d(self.mu_q_CFL_with_kaons_vec[self.mix_index_start:],
                                        self.e_CFL_with_kaons_vec[self.mix_index_start:],
                                        kind="linear",fill_value="extrapolate")


        try:
            self.P_of_mu_q_kaons = interp1d(self.mu_q_CFL_with_kaons_vec[self.mix_index_start:],
                                        self.P_CFL_with_kaons_vec[self.mix_index_start:],
                                        kind="quadratic",fill_value="extrapolate")
        except:
            self.P_of_mu_q_kaons = interp1d(self.mu_q_CFL_with_kaons_vec[self.mix_index_start:],
                                        self.P_CFL_with_kaons_vec[self.mix_index_start:],
                                        kind="linear",fill_value="extrapolate")


        try:
            self.rho_of_mu_q_kaons = interp1d(self.mu_q_CFL_with_kaons_vec[self.mix_index_start:],
                                        self.rho_CFL_with_kaons_vec[self.mix_index_start:],
                                        kind="quadratic",fill_value="extrapolate")
        except:
            self.rho_of_mu_q_kaons = interp1d(self.mu_q_CFL_with_kaons_vec[self.mix_index_start:],
                                        self.rho_CFL_with_kaons_vec[self.mix_index_start:],
                                        kind="linear",fill_value="extrapolate")

        return

    #Function that returns zero when electron chemical potential is equal in neutron matter and the mixed phase
    def find_transition_mu_q_to_mixphase_functional(self,mu_q):
        mu_q_NM_vec = self.eos_low_dens.mu_n_vec/3
        mu_e_NM_vec = self.eos_low_dens.mu_e_vec
        return self.mu_e_of_mu_q_kaons(mu_q)-np.interp(mu_q,mu_q_NM_vec,mu_e_NM_vec)

    #Finds the transition density from neutron matter to mixed phase by solving for zeros of find_transition_mu_q_to_mixphase_functional
    def find_transition_mu_q_to_mixphase(self,mu_q_guess):
        mu_q_guess_temp = mu_q_guess
        for i in range(10):
            try:
                mu_q_intersect,info,status,message = optimize.fsolve(self.find_transition_mu_q_to_mixphase_functional,mu_q_guess,full_output =
                    True)
            except:
                self.status = "no CFL"
                return 0

            if(status!=1):
                print(message)
                if(np.shape(mu_q_guess == ())):
                    mu_q_guess = mu_q_guess+mu_q_guess_temp*random.randrange(-10,10)/100
                else:
                    mu_q_guess = [mu_q_guess_temp[0]+mu_q_guess_temp[0]*random.randrange(-10,10)/100]
            else:
                break
        if(i==9):
            self.status = "no mix"
        return mu_q_intersect[0]



    #Compute full equation of state with Apr03 or RMF at low density, kaon with CFL mixed phase, and pure CFL
    def full_eos(self):
        if(self.status!="Success"):
            print("Possible issue. Status:"+str(self.status))
        self.mu_q_vec = np.linspace(self.mu_q_min,self.mu_q_max,self.N)
        self.P_vec = np.zeros(self.N)
        self.e_vec = np.zeros(self.N)
        self.rho_vec = np.zeros(self.N)
        self.mu_e_vec = np.zeros(self.N)

        #If the kaon phase has in any way failed,
        #return the low density eos


        #If there is a mixed phase, find the point where the transition happens
        if(self.mix_phase == True and self.status == "Success"):
            actual_transition_index = self.mix_index_start
            self.mu_q_transition = self.find_transition_mu_q_to_mixphase([self.mu_q_CFL_with_kaons_vec[self.mix_index_start-1]])

        if(self.status == "Fail" or self.status == "no CFL"):
            self.P_vec = self.eos_low_dens.P_vec
            self.e_vec = self.eos_low_dens.e_vec
            self.rho_vec = self.eos_low_dens.rho_vec
            self.mu_q_vec = self.eos_low_dens.mu_n_vec/3
            print("Possible issue. Status:"+str(self.status))
            return

        #If there is not a mixed phase (which either happens by design
        #or when solving for the transition position fails), then
        #only compute transition from low density phase to pure CFL
        if(self.status=="no mix" or self.mix_phase==False):
            for i in range(self.N):
                mu_q = self.mu_q_vec[i]
                mu_q_NM_vec = self.eos_low_dens.mu_n_vec/3
                rho_NM = np.interp(mu_q,mu_q_NM_vec,self.eos_low_dens.rho_vec)

                e_NM = np.interp(mu_q,mu_q_NM_vec,self.eos_low_dens.e_vec)
                P_NM = np.interp(mu_q,mu_q_NM_vec,self.eos_low_dens.P_vec)
                rho_CFL = np.interp(mu_q,self.mu_q_CFL_vec,self.rho_CFL_vec)
                e_CFL = np.interp(mu_q,self.mu_q_CFL_vec,self.e_CFL_vec)
                P_CFL = np.interp(mu_q,self.mu_q_CFL_vec,self.P_CFL_vec)
                mu_e_NM = np.interp(mu_q,mu_q_NM_vec,self.eos_low_dens.mu_e_vec)
                mu_e_CFL = 0
                if(P_NM>P_CFL):
                    self.P_vec[i] = P_NM
                    self.rho_vec[i] = rho_NM
                    self.e_vec[i] = e_NM
                else:
                    self.rho_vec[i] = rho_CFL
                    self.e_vec[i] = e_CFL
                    self.P_vec[i] = P_CFL
            return



        else:
            #If there is a mixed phase, compute full EoS
            #We start in neutron matter phase (MIX=False)
            MIX = False
            for i in range(self.N):
                mu_q = self.mu_q_vec[i]


                mu_q_NM_vec = self.eos_low_dens.mu_n_vec/3
                rho_NM = np.interp(mu_q,mu_q_NM_vec,self.eos_low_dens.rho_vec)

                e_NM = np.interp(mu_q,mu_q_NM_vec,self.eos_low_dens.e_vec)
                P_NM = np.interp(mu_q,mu_q_NM_vec,self.eos_low_dens.P_vec)
                mu_e_NM = np.interp(mu_q,mu_q_NM_vec,self.eos_low_dens.mu_e_vec)

                rho_mixed = self.rho_of_mu_q_kaons(mu_q)
                e_mixed = self.e_of_mu_q_kaons(mu_q)
                P_mixed = self.P_of_mu_q_kaons(mu_q)
                mu_e_mixed = self.mu_e_of_mu_q_kaons(mu_q)

                #If the electron chemical potential in the mixed phase
                #crosses that in the low density phase, we enter the mixed phase,
                #setting MIX=True
                if(mu_e_mixed>0 and mu_e_mixed<mu_e_NM and MIX==False):
                    MIX = True
                    actual_transition_index = i

                #Check that conditions to stay in the mixed phase still holds.
                if(mu_e_mixed<mu_e_NM and MIX==True):
                    self.rho_vec[i] = rho_mixed
                    self.P_vec[i] = P_mixed
                    self.e_vec[i] = e_mixed
                    self.mu_e_vec[i] = mu_e_mixed
                else:
                    self.rho_vec[i] = rho_NM
                    self.P_vec[i] = P_NM
                    self.e_vec[i] = e_NM
                    self.mu_e_vec[-i] = mu_e_NM

            #Add exact transition point to mix phase
            self.mu_q_vec[actual_transition_index] = self.mu_q_transition
            self.P_vec[actual_transition_index]=np.interp(self.mu_q_transition,self.eos_low_dens.mu_n_vec/3,self.eos_low_dens.P_vec)
            self.e_vec[actual_transition_index]=np.interp(self.mu_q_transition,self.eos_low_dens.mu_n_vec/3,self.eos_low_dens.e_vec)
            self.rho_vec[actual_transition_index]=np.interp(self.mu_q_transition,self.eos_low_dens.mu_n_vec/3,self.eos_low_dens.rho_vec)

        #The code below joins the low density and the high density phases
        #together around the transition point found above.

        self.e_vec = self.e_vec[actual_transition_index:]
        self.rho_vec = self.rho_vec[actual_transition_index:]
        self.mu_q_vec = self.mu_q_vec[actual_transition_index:]
        self.P_vec = self.P_vec[actual_transition_index:]


        p=self.eos_low_dens.P_vec[0]
        i=0
        p_min = self.P_vec[0]
        P_add = []
        e_add = []
        mu_q_add = []
        rho_add = []
        while(p<p_min):
            P_add.append(p)
            e = self.eos_low_dens.e_vec[i]
            e_add.append(e)
            mu_q = self.eos_low_dens.mu_n_vec[i]/3
            mu_q_add.append(mu_q)
            rho = self.eos_low_dens.rho_vec[i]
            rho_add.append(rho)
            i+=1
            p=self.eos_low_dens.P_vec[i]

        self.P_vec = np.concatenate((np.array(P_add),self.P_vec))
        self.e_vec = np.concatenate((np.array(e_add),self.e_vec))
        self.rho_vec = np.concatenate((np.array(rho_add),self.rho_vec))
        self.mu_q_vec = np.concatenate((np.array(mu_q_add),self.mu_q_vec))
        if(self.status!="Success"):
            print("Possible issue. Status:"+str(self.status))
        return

    def add_crust(self):
        P_conv = 8.96057e-7 #Convert between MeV/fm^3 to M_sol/km^3
        e_temp = []
        P_temp = []
        rho_temp = []
        for line in reversed(list(open(self.filename_crust,'r'))):
            e,P,rho = line.split()[:3]
            e = np.float64(e)/P_conv
            P = np.float64(P)/P_conv
            rho = np.float64(rho)
            if(P<self.P_vec[0]):
                e_temp.append(e)
                P_temp.append(P)
                rho_temp.append(rho)
            else:
                break

        self.e_w_crust_vec = np.concatenate((np.array(e_temp),np.array(self.e_vec)))
        self.P_w_crust_vec = np.concatenate((np.array(P_temp),np.array(self.P_vec)))
        self.rho_w_crust_vec = np.concatenate((np.array(rho_temp),np.array(self.rho_vec)))



