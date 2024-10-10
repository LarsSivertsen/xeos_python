import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import random
import TOV_Rahul
from scipy.interpolate import interp1d

class apr03_EoS(object):
    def __init__ (self,N=100
                      ,rho_min = 0.1
                      ,rho_max = 2
                      ,d_rho = 0.0001
                      ,tol = 1e-6):

        #initiate all constants that are not input from user
        self.constants()

        #Number of points used in APR03 eos
        self.N = N

        #Lowest density APR eos starts from
        self.rho_min = rho_min

        #Highest density APR eos integrates to
        self.rho_max = rho_max

        #Step size number denisty (rho) used for computing chemical potentials
        self.d_rho = d_rho

        #Absolute tolerance used in solvers
        self.tol = tol

        self.build_PNM_and_SNM()            #Build apr03 EoS with pure nuclear matter (PNM) and symmetric nuclear matter (SNM)
                                                  #in the low density phase (LDP) and high density phase (HDP). Creates the instances
                                                  #rho_vec (total number density in fm^-3)
                                                  #e_PNM_LDP (energy density for PNM in the LDP)
                                                  #e_SNM_LDP (energy density for SNM in the LDP)
                                                  #e_PNM_HDP (energy density for PNM in the HDP)
                                                  #e_SNM_HDP (energy density for SNM in the HDP)

        #Build full APR neutron matter in LDP and HDP phase
        self.build_full_LDP_HDP_EoS()
        #Find the transition density between LDP and HDP in APR03
        self.LDP_HDP_transition()
        #Make a connected APR03 EoS (Going from LDP to HDP)
        self.build_connected_EoS()

    def constants(self):
        #Physical consants
        self.m = 938.92         #Average mass neutron and proton in MeV
        self.m_mu = 105.66      #Mass of muon in MeV
        self.m_e = 0.50998      #Mass of electron in MeV

        #Conversion constants
        self.hc = 197.327       #Plancks reduced constant hbar times the speed of light in MeV fm

        ##### apr03 constants ####
        self.p1 =  337.2
        self.p2 = -382.0
        self.p3 = 89.8
        self.p4 = 0.457
        self.p5 =-59.0
        self.p6 = -19.1
        self.p7 = 214.6
        self.p8 = -384.0
        self.p9 = 6.4
        self.p10 = 69.0
        self.p11 = -33.0
        self.p12 = 0.35
        self.p13 = 0.0
        self.p14 = 0.0
        self.p15 = 287.0
        self.p16 = -1.54
        self.p17 = 175.0
        self.p18 = -1.45
        self.p19 = 0.32
        self.p20 = 0.195
        self.p21 = 0.0




    def e_functional(self,rho,xp):
        #eq 2 in xeos note
        if(rho*(1-xp)<0 or rho*xp<0):
            tau_n = 0
            tau_p = 0
        else:
            tau_n = 1/(5*np.pi**2)*(3*np.pi**2*rho*(1-xp))**(5/3)
            tau_p = 1/(5*np.pi**2)*(3*np.pi**2*rho*xp)**(5/3)

        #eq 1 in xeos note
        if(self.p4*rho>200):
            e = 1/(2*self.m)*self.hc**2*tau_n
            e += 1/(2*self.m)*self.hc**2*tau_p
        else:
            e = (1/(2*self.m)*self.hc**2+(self.p3+(1-xp)*self.p5)*rho*np.exp(-self.p4*rho))*tau_n
            e += (1/(2*self.m)*self.hc**2+(self.p3+xp*self.p5)*rho*np.exp(-self.p4*rho))*tau_p

        #gLn = gL(rho,xp=0), eq 3 in xeos note
        #gLs = gL(rho,xp=0.5), eq 3 in xeos note
        #(If statment is to avoid issues with overflow errors in large exponentials)
        if(self.p9**2*rho**2>200):
            gLn = -rho*(self.p12+self.p7*rho+self.p8*rho**2)

            gLs = -rho**2*(self.p1+self.p2*rho+self.p6*rho**2)
        else:

            gLn = -rho*(self.p12+self.p7*rho+self.p8*rho**2+self.p13*rho*np.exp(-self.p9**2*rho**2))

            gLs = -rho**2*(self.p1+self.p2*rho+self.p6*rho**2+(self.p10+self.p11*rho)*np.exp(-self.p9**2*rho**2))

        #(Again, if statment just avoids large exponentials)
        if(self.p16*(rho-self.p20)>200):
            gHn = gLn-np.exp(200)

        elif(self.p16*(rho-self.p20)>-200):
            #gH(rho,xp=0), eq 4 in xeos note
            gHn = gLn-rho**2*(self.p15*(rho-self.p20)+self.p14*(rho-self.p20)**2)*np.exp(self.p16*(rho-self.p20))
        else:
            gHn = gLn

        #gH(rho,xp-0.5), eq 4 in xeos note
        gHs = gLs-rho**2*(self.p17*(rho-self.p19)+self.p21*(rho-self.p19)**2)*np.exp(self.p18*(rho-self.p19))

        #energy density low density phase (LDP)
        eLDP = e+gLs*(1-(1-2*xp)**2)+gLn*(1-2*xp)**2

        #energy density high density phase (HDP)
        eHDP = e+gHs*(1-(1-2*xp)**2)+gHn*(1-2*xp)**2
        return eLDP+self.m*rho,eHDP+self.m*rho

    #Create instances for number density,  energy density, and energy per nucleon
    #for the LDP and HDP of pure nuclear matter (PNM) and symmetric nuclear matter (SNM)
    def build_PNM_and_SNM(self):
        #Number density for both LDP and HDP
        self.rho_PNM_SNM_vec = np.linspace(self.rho_min,self.rho_max,self.N)

        #energy density pure neutron matter (PNM) in the LDP
        self.e_PNM_LDP_vec = np.zeros(self.N)

        #energy density PNM in the HDP
        self.e_PNM_HDP_vec = np.zeros(self.N)

        #energy density symmetric nuclear matter (SNM) in the LDP
        self.e_SNM_LDP_vec = np.zeros(self.N)

        #energy density SNM in the HDP
        self.e_SNM_HDP_vec = np.zeros(self.N)

        for i in range(self.N):
            self.e_PNM_LDP_vec[i],self.e_PNM_HDP_vec[i] = self.e_functional(self.rho_PNM_SNM_vec[i],0)
            self.e_SNM_LDP_vec[i],self.e_SNM_HDP_vec[i] = self.e_functional(self.rho_PNM_SNM_vec[i],0.5)

        #Compute the energy per nucleon for all combiniations of PNM, SNM, LDP, HDP
        self.E_per_A_PNM_LDP_vec = self.e_PNM_LDP_vec/self.rho_PNM_SNM_vec-self.m
        self.E_per_A_PNM_HDP_vec = self.e_PNM_HDP_vec/self.rho_PNM_SNM_vec-self.m
        self.E_per_A_SNM_LDP_vec = self.e_SNM_LDP_vec/self.rho_PNM_SNM_vec-self.m
        self.E_per_A_SNM_HDP_vec = self.e_SNM_HDP_vec/self.rho_PNM_SNM_vec-self.m
        return

    #Nucleon chemical potentials as a function of density rho and proton fraction xp
    def chemical_potential_functional(self,rho,xp,LDP_HDP):
        #energy density for input rho+d_rho, xp
        e_forward_rho = self.e_functional(rho+self.d_rho,xp)[LDP_HDP]

        #stepsize in proton fraction xp
        d_xp = self.d_rho*xp

        #energy density for rho, xp+d_xp
        e_forward_xp = self.e_functional(rho,xp+d_xp)[LDP_HDP]

        #Check if d_rho is larger that rho. If so, symmetric derivative is used
        #else, forward derivative is used
        if(rho>self.d_rho):
            e_backward_rho = self.e_functional(rho-self.d_rho,xp)[LDP_HDP]
            d_rho_full = 2*self.d_rho
        else:
            e_backward_rho = self.e_functional(rho,xp)[LDP_HDP]
            d_rho_full = self.d_rho

        #Check if xp is larger than d_xp. If so symmetric derivative is used
        #else, forward derivative is used
        if(xp>d_xp):
            e_backward_xp = self.e_functional(rho,xp-d_xp)[LDP_HDP]
            d_xp_full = 2*d_xp
        else:
            e_backward_xp = self.e_functional(rho,xp)[LDP_HDP]
            d_xp_full = d_xp


        #Chemcial potential for neutron (mu_n) and proton (mu_p)
        #mu_n = de/d_rho-xp/rho*de/d_xp
        #mu_p = de/d_rho+(1-xp)/rho*de/d_xp
        mu_n = (e_forward_rho-e_backward_rho)/(d_rho_full)-(xp/rho)*(e_forward_xp-e_backward_xp)/(d_xp_full)
        mu_p = (e_forward_rho-e_backward_rho)/(d_rho_full)+((1-xp)/rho)*(e_forward_xp-e_backward_xp)/(d_xp_full)
        return mu_n, mu_p

    #Lepton energy density for chemical potential mu and lepton mass m_l
    def lepton_energy_density(self,mu,m_l):
        if(mu<m_l):
            return 0
        pF = np.sqrt(mu**2-m_l**2)
        return ((pF*mu**3)+(pF**3*mu)+m_l**4*np.log(mu/(pF+mu)))/(8*np.pi**2*self.hc**3)

    #Total lepton energy density as a function of density rho, proton fraction xp
    #and bool: LDP_HDP=0 in LDP, =1 in HDP. Assuming beta equilibrium
    def total_lepton_energy_density(self,rho,xp,LDP_HDP):
        #chemical potentials mu_n and mu_p
        mu_n,mu_p = self.chemical_potential_functional(rho,xp,LDP_HDP)

        #Beta equilibrium gives the electron chemical potential mu_e=mu_n-mu_p, and
        #muon chemical potenital mu_e=mu_mu
        mu_e = mu_n-mu_p

        #Compute electron and muon energy densities
        e_e = self.lepton_energy_density(mu_e,self.m_e)
        e_mu = self.lepton_energy_density(mu_e,self.m_mu)
        return e_e+e_mu

    #Total energy density functional, this function is used to determine
    #the proton fraction xp as a function of number density rho
    def total_energy_density_functional(self,rho,xp,LDP_HDP):
        #total lepton energy density
        e_l = self.total_lepton_energy_density(rho,xp,LDP_HDP)

        #total nuclear energy density
        e_nuclear = self.e_functional(rho,xp)[LDP_HDP]
        return e_l+e_nuclear

    #When computing the mixed phase of nuclear matter and CFL, we need the
    #energy density as a function of electron chemical potential.
    def total_energy_density_of_mu_e(self,rho,xp,mu_n,mu_p,mu_e,LDP_HDP):
        e_nuclear = self.e_functional(rho,xp)[LDP_HDP]
        e_e = self.lepton_energy_density(mu_e,self.m_e)
        e_mu = self.lepton_energy_density(mu_e,self.m_mu)
        return e_e+e_mu+e_nuclear




    def total_pressure_of_mu_e(self,mu_n,mu_e,xp,rho,LDP_HDP):
        #Proton chemcial potential is given by beta equilibrium condition
        mu_p = mu_n-mu_e

        #compute electron and muon fermi momenta pF_e, and pF_mu
        if(mu_e>=self.m_e):
            pF_e = np.sqrt(mu_e**2-self.m_e**2)
        else:
            pF_e = 0

        if(mu_e>=self.m_mu):
            pF_mu = np.sqrt(mu_e**2-self.m_mu**2)
        else:
            pF_mu = 0

        rho_e = pF_e**3/(3*np.pi**2*self.hc**3)
        rho_mu = pF_mu**3/(3*np.pi**2*self.hc**3)
        return (mu_n*(1-xp)*rho+mu_p*xp*rho+mu_e*rho_e+mu_e*rho_mu
               -self.total_energy_density_of_mu_e(rho,xp,mu_n,mu_p,mu_e,LDP_HDP))





    #This function retruns total charge density of protons, electrons and muons
    #We use this to determine xp at a given rho when we are in the nuclear matter phase
    def find_xp_functional(self,xp,rho,LDP_HDP):
        mu_n,mu_p = self.chemical_potential_functional(rho,xp,LDP_HDP)
        mu_e = mu_n-mu_p
        if(mu_e>=self.m_e):
            pF_e = np.sqrt(mu_e**2-self.m_e**2)
        else:
            pF_e = 0

        if(mu_e>=self.m_mu):
            pF_mu = np.sqrt(mu_e**2-self.m_mu**2)
        else:
            pF_mu = 0

        rho_e = pF_e**3/(3*np.pi**2*self.hc**3)
        rho_mu = pF_mu**3/(3*np.pi**2*self.hc**3)
        rho_p = xp*rho

        return rho_e+rho_mu-rho_p

    #Solve for the zeros of find_xp_functional to find xp in the nuclear matter phase
    def find_xp(self,rho,xp_guess,LDP_HDP):
        for i in range(10):
            xp,info,status,message= optimize.fsolve(self.find_xp_functional,xp_guess,args=(rho,LDP_HDP),full_output = True)
            if(message!="The solution converged."):
                if(np.shape(xp_guess)==()):
                    xp_guess += random.randrange(-10,10)/100*xp_guess
                else:
                    xp_guess[0] += random.randrange(-10,10)/100*xp_guess[0]
            else:
                return xp
        if(abs(self.find_xp_functional(xp,rho,LDP_HDP))>self.tol):
            print("Solver find_xp failed to find xp")
        return xp

    #Build the full low density (LDP) and high density (HDP) phase of APR03 for neutron matter
    def build_full_LDP_HDP_EoS(self):
        self.rho_LDP_HDP_vec = np.linspace(self.rho_min,self.rho_max,self.N)
        self.xp_LDP_vec = np.zeros(self.N)
        self.e_LDP_vec = np.zeros(self.N)
        self.mu_n_LDP_vec = np.zeros(self.N)
        self.mu_p_LDP_vec = np.zeros(self.N)
        self.xp_LDP_guess = 0.01

        self.xp_HDP_vec = np.zeros(self.N)
        self.e_HDP_vec = np.zeros(self.N)
        self.mu_n_HDP_vec = np.zeros(self.N)
        self.mu_p_HDP_vec = np.zeros(self.N)
        self.xp_HDP_guess = 0.01

        for i in range(self.N):
            self.xp_LDP_vec[i] = self.find_xp(self.rho_LDP_HDP_vec[i],self.xp_LDP_guess,0)

            self.e_LDP_vec[i] = self.total_energy_density_functional(self.rho_LDP_HDP_vec[i],self.xp_LDP_vec[i],0)

            self.mu_n_LDP_vec[i],self.mu_p_LDP_vec[i] = (
                    self.chemical_potential_functional(self.rho_LDP_HDP_vec[i],self.xp_LDP_vec[i],0))

            xp_LDP_guess = self.xp_LDP_vec[i]

            self.xp_HDP_vec[i] = self.find_xp(self.rho_LDP_HDP_vec[i],self.xp_HDP_guess,1)
            self.e_HDP_vec[i] = self.total_energy_density_functional(self.rho_LDP_HDP_vec[i],self.xp_HDP_vec[i],1)

            self.mu_n_HDP_vec[i],self.mu_p_HDP_vec[i] = (
                    self.chemical_potential_functional(self.rho_LDP_HDP_vec[i],self.xp_HDP_vec[i],1))

            xp_HDP_guess = self.xp_HDP_vec[i]

        self.E_per_A_LDP_vec = self.e_LDP_vec/self.rho_LDP_HDP_vec-self.m
        self.E_per_A_HDP_vec = self.e_HDP_vec/self.rho_LDP_HDP_vec-self.m

        self.mu_e_LDP_vec = self.mu_n_LDP_vec-self.mu_p_LDP_vec
        self.pF_e_LDP_vec = np.heaviside(self.mu_e_LDP_vec-self.m_e,0)*np.sqrt(abs(self.mu_e_LDP_vec**2-self.m_e**2))
        self.rho_e_LDP_vec = self.pF_e_LDP_vec**3/(3*np.pi**2*self.hc**3)

        self.pF_mu_LDP_vec = np.heaviside(self.mu_e_LDP_vec-self.m_mu,0)*np.sqrt(abs(self.mu_e_LDP_vec**2-self.m_mu**2))
        self.rho_mu_LDP_vec = self.pF_mu_LDP_vec**3/(3*np.pi**2*self.hc**3)

        self.mu_e_HDP_vec = self.mu_n_HDP_vec-self.mu_p_HDP_vec
        self.pF_e_HDP_vec = np.heaviside(self.mu_e_HDP_vec-self.m_e,0)*np.sqrt(abs(self.mu_e_HDP_vec**2-self.m_e**2))
        self.rho_e_HDP_vec = self.pF_e_HDP_vec**3/(3*np.pi**2*self.hc**3)

        self.pF_mu_HDP_vec = np.heaviside(self.mu_e_HDP_vec-self.m_mu,0)*np.sqrt(abs(self.mu_e_HDP_vec**2-self.m_mu**2))
        self.rho_mu_HDP_vec = self.pF_mu_HDP_vec**3/(3*np.pi**2*self.hc**3)

        self.E_per_A_LDP_with_lepton_vec = (self.E_per_A_LDP_vec
                                           +self.mu_e_LDP_vec*(self.rho_e_LDP_vec+self.rho_mu_LDP_vec))

        self.E_per_A_HDP_with_lepton_vec = (self.E_per_A_HDP_vec
                                           +self.mu_e_HDP_vec*(self.rho_e_HDP_vec+self.rho_mu_HDP_vec))



        self.P_LDP_vec = (self.mu_n_LDP_vec*self.rho_LDP_HDP_vec*(1-self.xp_LDP_vec)
                               +self.mu_p_LDP_vec*self.rho_LDP_HDP_vec*self.xp_LDP_vec
                               +self.mu_e_LDP_vec*self.rho_e_LDP_vec
                               +self.mu_e_LDP_vec*self.rho_mu_LDP_vec
                               -self.e_LDP_vec)

        self.P_HDP_vec = (self.mu_n_HDP_vec*self.rho_LDP_HDP_vec*(1-self.xp_HDP_vec)
                               +self.mu_p_HDP_vec*self.rho_LDP_HDP_vec*self.xp_HDP_vec
                               +self.mu_e_HDP_vec*self.rho_e_HDP_vec
                               +self.mu_e_HDP_vec*self.rho_mu_HDP_vec
                               -self.e_HDP_vec)
        return

    #Functional used to find the density at which the transition from LDP to HDP starts
    #This is the density the energy per nucleon is equal in both LDP and HDP
    def LDP_HDP_intersect_rho_functional(self,rho):
        return (np.interp(rho,self.rho_LDP_HDP_vec,self.E_per_A_LDP_with_lepton_vec)
                -np.interp(rho,self.rho_LDP_HDP_vec,self.E_per_A_HDP_with_lepton_vec))

    #Functional used to find the density at which the transition from LDP to HDP ends
    #This is where the pressure in HDP is equal to that of LDP in the beginning of
    #the transition
    def LDP_HDP_intersect_P_functional(self,rho):
        return (np.interp(rho,self.rho_LDP_HDP_vec,self.P_HDP_vec)
                -np.interp(self.rho_LDP_intersect,self.rho_LDP_HDP_vec,self.P_LDP_vec))

    #Function that gives the two densities at which the transition between LDP and HDP happens
    def LDP_HDP_transition(self):
        self.rho_LDP_transition_index = 0
        for i in range(self.N):
            if(self.E_per_A_LDP_with_lepton_vec[i]>self.E_per_A_HDP_with_lepton_vec[i]):
                self.rho_LDP_intersect = optimize.fsolve(
                        self.LDP_HDP_intersect_rho_functional,[self.rho_LDP_HDP_vec[i]])
                self.rho_LDP_transition_index = i
                break

        if(abs(self.LDP_HDP_intersect_rho_functional(self.rho_LDP_intersect)>self.tol)):
            print("LDP_HDP_transition failed to find initial density")

        for i in range(self.rho_LDP_transition_index,self.N):
            if(self.P_HDP_vec[i]>self.P_LDP_vec[self.rho_LDP_transition_index]):
                rho_guess = [self.rho_LDP_HDP_vec[i]+0.2]
                for j in range(10):
                    rho,info,status,message= optimize.fsolve(
                        self.LDP_HDP_intersect_P_functional,rho_guess,full_output = True)
                    if(status!=1):
                        rho_guess = [rho_guess[0]+random.randrange(-10,10)/100]
                    else:
                        break
                self.rho_HDP_intersect = rho
                self.rho_HDP_transition_index = i
                break

        if(abs(self.LDP_HDP_intersect_P_functional(self.rho_HDP_intersect))>self.tol):
            print("LDP_HDP_transition failed to find end density")



    #Using the transition densities and the full APR LDP and HDP, compute the connected APR03 EoS
    def build_connected_EoS(self):
        self.N_combined = self.N-(self.rho_HDP_transition_index-self.rho_LDP_transition_index)+2
        self.rho_vec = np.zeros(self.N_combined)
        self.xp_vec = np.zeros(self.N_combined)
        self.e_vec = np.zeros(self.N_combined)
        self.mu_n_vec = np.zeros(self.N_combined)
        self.mu_p_vec = np.zeros(self.N_combined)


        #Everything from here to ##### is connecting the EoS LDP to HDP at the transtition
        self.rho_vec[:self.rho_LDP_transition_index] = self.rho_LDP_HDP_vec[:self.rho_LDP_transition_index]
        self.rho_vec[self.rho_LDP_transition_index+2:] = self.rho_LDP_HDP_vec[self.rho_HDP_transition_index:]
        self.rho_vec[self.rho_LDP_transition_index] = self.rho_LDP_intersect
        self.rho_vec[self.rho_LDP_transition_index+1] = self.rho_HDP_intersect


        self.xp_vec[:self.rho_LDP_transition_index] = self.xp_LDP_vec[:self.rho_LDP_transition_index]
        self.xp_vec[self.rho_LDP_transition_index+2:] = self.xp_HDP_vec[self.rho_HDP_transition_index:]
        self.xp_vec[self.rho_LDP_transition_index] = (
                self.find_xp(self.rho_LDP_intersect,self.xp_LDP_vec[self.rho_LDP_transition_index],0))
        self.xp_vec[self.rho_LDP_transition_index+1] = (
                self.find_xp(self.rho_HDP_intersect,self.xp_HDP_vec[self.rho_HDP_transition_index],1))

        self.e_vec[:self.rho_LDP_transition_index] = self.e_LDP_vec[:self.rho_LDP_transition_index]
        self.e_vec[self.rho_LDP_transition_index+2:] = self.e_HDP_vec[self.rho_HDP_transition_index:]
        self.e_vec[self.rho_LDP_transition_index] = (
                self.total_energy_density_functional(self.rho_LDP_intersect,self.xp_vec[self.rho_LDP_transition_index],0))
        self.e_vec[self.rho_LDP_transition_index+1] = (
                self.total_energy_density_functional(self.rho_HDP_intersect,self.xp_vec[self.rho_LDP_transition_index+1],1))

        self.mu_n_vec[:self.rho_LDP_transition_index] = self.mu_n_LDP_vec[:self.rho_LDP_transition_index]
        self.mu_n_vec[self.rho_LDP_transition_index+2:] = self.mu_n_HDP_vec[self.rho_HDP_transition_index:]

        self.mu_n_vec[self.rho_LDP_transition_index],self.mu_p_vec[self.rho_LDP_transition_index] = (
                self.chemical_potential_functional(self.rho_LDP_intersect,self.xp_vec[self.rho_LDP_transition_index],0))
        self.mu_n_vec[self.rho_LDP_transition_index+1],self.mu_p_vec[self.rho_LDP_transition_index+1] = (
                self.chemical_potential_functional(self.rho_HDP_intersect,self.xp_vec[self.rho_LDP_transition_index+1],1))

        self.mu_p_vec[:self.rho_LDP_transition_index] = self.mu_p_LDP_vec[:self.rho_LDP_transition_index]
        self.mu_p_vec[self.rho_LDP_transition_index+2:] = self.mu_p_HDP_vec[self.rho_HDP_transition_index:]
        #####


        self.mu_e_vec = self.mu_n_vec-self.mu_p_vec
        self.kFe_vec = np.heaviside(self.mu_e_vec-self.m_e,0)*np.sqrt(self.mu_e_vec**2-self.m_e**2)
        self.rho_e_vec = self.kFe_vec**3/(3*np.pi**2*self.hc**3)

        self.kFmu_vec = np.heaviside(self.mu_e_vec-self.m_mu,0)*np.sqrt(abs(self.mu_e_vec**2-self.m_mu**2))
        self.rho_mu_vec = self.kFmu_vec**3/(3*np.pi**2*self.hc**3)

        self.P_vec = (self.mu_n_vec*self.rho_vec*(1-self.xp_vec)
                                        +self.mu_p_vec*self.rho_vec*self.xp_vec
                                        +self.mu_e_vec*self.rho_e_vec
                                        +self.mu_e_vec*self.rho_mu_vec
                                        -self.e_vec)

        return


