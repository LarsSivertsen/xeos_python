import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import random
import TOV_Rahul

class apr03_EoS(object):
    def __init__ (self,N_apr03
                      ,rho_min_apr03=0.1
                      ,rho_max_apr03 = 2
                      ,d_rho = 0.0001
                      ,tol = 1e-6):

        #initiate all constants that are not input from user
        self.constants()
        
        #Number of points used in APR03 eos
        self.N_apr03 = N_apr03
        
        #Lowest density APR eos starts from 
        self.rho_min_apr03 = rho_min_apr03
        
        #Highest density APR eos integrates to 
        self.rho_max_apr03 = rho_max_apr03
        
        #Step size number denisty (rho) used for computing chemical potentials
        self.d_rho = d_rho
        
        #Absolute tolerance used in solvers
        self.tol = tol
        
        self.build_apr03_PNM_and_SNM()            #Build apr03 EoS with pure nuclear matter (PNM) and symmetric nuclear matter (SNM)
                                                  #in the low density phase (LDP) and high density phase (HDP). Creates the instances
                                                  #rho_vec (total number density in fm^-3)
                                                  #e_PNM_LDP (energy density for PNM in the LDP)
                                                  #e_SNM_LDP (energy density for SNM in the LDP)
                                                  #e_PNM_HDP (energy density for PNM in the HDP)
                                                  #e_SNM_HDP (energy density for SNM in the HDP)
        
        #Build full APR neutron matter in LDP and HDP phase
        self.build_full_LDP_HDP_apr03_EoS()
        #Find the transition density between LDP and HDP in APR03
        self.LDP_HDP_transition()
        #Make a connected APR03 EoS (Going from LDP to HDP)
        self.build_connected_apr03_EoS()

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

        


    def apr03_functional(self,rho,x_p):
        #if(rho<0):
        #    return self.apr03_functional(0,x_p)+1e10*((rho)**2)
    
        #eq 2 in xeos note
        if(rho*(1-x_p)<0 or rho*x_p<0):
            tau_n = 0
            tau_p = 0
        else:
            tau_n = 1/(5*np.pi**2)*((3*np.pi**2*rho*(1-x_p)))**(5/3)
            tau_p = 1/(5*np.pi**2)*((3*np.pi**2*rho*x_p))**(5/3)
        
        #eq 1 in xeos note
        if(self.p4*rho>200):
            e = 1/(2*self.m)*self.hc**2*tau_n
            e += 1/(2*self.m)*self.hc**2*tau_p
        else:
            e = (1/(2*self.m)*self.hc**2+(self.p3+(1-x_p)*self.p5)*rho*np.exp(-self.p4*rho))*tau_n
            e += (1/(2*self.m)*self.hc**2+(self.p3+x_p*self.p5)*rho*np.exp(-self.p4*rho))*tau_p
       
       
        #gLn = gL(rho,x_p=0), eq 3 in xeos note 
        #gLs = gL(rho,x_p=0.5), eq 3 in xeos note
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
            #gH(rho,x_p=0), eq 4 in xeos note
            gHn = gLn-rho**2*(self.p15*(rho-self.p20)+self.p14*(rho-self.p20)**2)*np.exp(self.p16*(rho-self.p20))
        else:
            gHn = gLn
        
    
        #gH(rho,x_p-0.5), eq 4 in xeos note
        gHs = gLs-rho**2*(self.p17*(rho-self.p19)+self.p21*(rho-self.p19)**2)*np.exp(self.p18*(rho-self.p19))
    
        #energy density low density phase (LDP)
        eLDP = e+gLs*(1-(1-2*x_p)**2)+gLn*(1-2*x_p)**2

        #energy density high density phase (HDP)
        eHDP = e+gHs*(1-(1-2*x_p)**2)+gHn*(1-2*x_p)**2
        return eLDP+self.m*rho,eHDP+self.m*rho

    #Create instances for number density,  energy density, and energy per nucleon 
    #for the LDP and HDP of pure nuclear matter (PNM) and symmetric nuclear matter (SNM)
    def build_apr03_PNM_and_SNM(self):
        #Number density for both LDP and HDP
        self.rho_PNM_SNM_apr03_vec = np.linspace(self.rho_min_apr03,self.rho_max_apr03,self.N_apr03)
        
        #energy density pure neutron matter (PNM) in the LDP 
        self.e_PNM_LDP_apr03_vec = np.zeros(self.N_apr03)
        
        #energy density PNM in the HDP
        self.e_PNM_HDP_apr03_vec = np.zeros(self.N_apr03)
        
        #energy density symmetric nuclear matter (SNM) in the LDP
        self.e_SNM_LDP_apr03_vec = np.zeros(self.N_apr03)
        
        #energy density SNM in the HDP
        self.e_SNM_HDP_apr03_vec = np.zeros(self.N_apr03)

        for i in range(self.N_apr03):
            self.e_PNM_LDP_apr03_vec[i],self.e_PNM_HDP_apr03_vec[i] = self.apr03_functional(self.rho_PNM_SNM_apr03_vec[i],0)
            self.e_SNM_LDP_apr03_vec[i],self.e_SNM_HDP_apr03_vec[i] = self.apr03_functional(self.rho_PNM_SNM_apr03_vec[i],0.5) 

        #Compute the energy per nucleon for all combiniations of PNM, SNM, LDP, HDP
        self.E_per_A_PNM_LDP_apr03_vec = self.e_PNM_LDP_apr03_vec/self.rho_PNM_SNM_apr03_vec-self.m
        self.E_per_A_PNM_HDP_apr03_vec = self.e_PNM_HDP_apr03_vec/self.rho_PNM_SNM_apr03_vec-self.m
        self.E_per_A_SNM_LDP_apr03_vec = self.e_SNM_LDP_apr03_vec/self.rho_PNM_SNM_apr03_vec-self.m
        self.E_per_A_SNM_HDP_apr03_vec = self.e_SNM_HDP_apr03_vec/self.rho_PNM_SNM_apr03_vec-self.m
        return
    
    #Nucleon chemical potentials as a function of density rho and proton fraction x_p
    def chemical_potential_apr03_functional(self,rho,x_p,LDP_HDP):
        #energy density for input rho+d_rho, x_p
        e_forward_rho = self.apr03_functional(rho+self.d_rho,x_p)[LDP_HDP]
        
        #stepsize in proton fraction x_p
        d_x_p = self.d_rho*x_p
        
        #energy density for rho, x_p+d_x_p
        e_forward_x_p = self.apr03_functional(rho,x_p+d_x_p)[LDP_HDP]
        
        #Check if d_rho is larger that rho. If so, symmetric derivative is used
        #else, forward derivative is used
        if(rho>self.d_rho):
            e_backward_rho = self.apr03_functional(rho-self.d_rho,x_p)[LDP_HDP]
            d_rho_full = 2*self.d_rho
        else:
            e_backward_rho = self.apr03_functional(rho,x_p)[LDP_HDP]
            d_rho_full = self.d_rho

        #Check if x_p is larger than d_x_p. If so symmetric derivative is used
        #else, forward derivative is used
        if(x_p>d_x_p):
            e_backward_x_p = self.apr03_functional(rho,x_p-d_x_p)[LDP_HDP]
            d_x_p_full = 2*d_x_p
        else:
            e_backward_x_p = self.apr03_functional(rho,x_p)[LDP_HDP]
            d_x_p_full = d_x_p

        
        #Chemcial potential for neutron (mu_n) and proton (mu_p)
        #mu_n = de/d_rho-x_p/rho*de/d_x_p
        #mu_p = de/d_rho+(1-x_p)/rho*de/d_x_p
        mu_n = (e_forward_rho-e_backward_rho)/(d_rho_full)-(x_p/rho)*(e_forward_x_p-e_backward_x_p)/(d_x_p_full)
        mu_p = (e_forward_rho-e_backward_rho)/(d_rho_full)+((1-x_p)/rho)*(e_forward_x_p-e_backward_x_p)/(d_x_p_full)
        return mu_n, mu_p

    #Lepton energy density for chemical potential mu and lepton mass m_l
    def lepton_energy_density_apr03(self,mu,m_l):
        if(mu<m_l):
            return 0
        pF = np.sqrt(mu**2-m_l**2)
        return ((pF*mu**3)+(pF**3*mu)+m_l**4*np.log(mu/(pF+mu)))/(8*np.pi**2*self.hc**3)
    
    #Total lepton energy density as a function of density rho, proton fraction x_p
    #and bool: LDP_HDP=0 in LDP, =1 in HDP. Assuming beta equilibrium
    def total_lepton_energy_density_apr03(self,rho,x_p,LDP_HDP):
        #chemical potentials mu_n and mu_p
        mu_n,mu_p = self.chemical_potential_apr03_functional(rho,x_p,LDP_HDP)
        
        #Beta equilibrium gives the electron chemical potential mu_e=mu_n-mu_p, and 
        #muon chemical potenital mu_e=mu_mu
        mu_e = mu_n-mu_p
        
        #Compute electron and muon energy densities
        e_e = self.lepton_energy_density_apr03(mu_e,self.m_e)
        e_mu = self.lepton_energy_density_apr03(mu_e,self.m_mu)
        return e_e+e_mu
    
    #Total energy density functional, this function is used to determine 
    #the proton fraction x_p as a function of number density rho
    def total_energy_density_functional_apr03(self,rho,x_p,LDP_HDP):
        #total lepton energy density
        e_l = self.total_lepton_energy_density_apr03(rho,x_p,LDP_HDP)
        
        #total nuclear energy density
        e_nuclear = self.apr03_functional(rho,x_p)[LDP_HDP]
        return e_l+e_nuclear
    
    #When computing the mixed phase of nuclear matter and CFL, we need the 
    #energy density as a function of electron chemical potential.
    def total_energy_density_of_mu_e_apr03(self,rho,x_p,mu_n,mu_p,mu_e,LDP_HDP):
        e_nuclear = self.apr03_functional(rho,x_p)[LDP_HDP]
        e_e = self.lepton_energy_density_apr03(mu_e,self.m_e)
        e_mu = self.lepton_energy_density_apr03(mu_e,self.m_mu)
        return e_e+e_mu+e_nuclear
    



    def total_pressure_of_mu_e_apr03_test(self,mu_n,mu_e,x_p,rho,LDP_HDP):
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
        return (mu_n*(1-x_p)*rho+mu_p*x_p*rho+mu_e*rho_e+mu_e*rho_mu
               -self.total_energy_density_of_mu_e_apr03(rho,x_p,mu_n,mu_p,mu_e,LDP_HDP))

        


        
    #This function retruns total charge density of protons, electrons and muons 
    #We use this to determine x_p at a given rho when we are in the nuclear matter phase
    def find_x_p_apr03_functional(self,x_p,rho,LDP_HDP):
        mu_n,mu_p = self.chemical_potential_apr03_functional(rho,x_p,LDP_HDP)
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
        rho_p = x_p*rho
        
        return rho_e+rho_mu-rho_p
    
    #Solve for the zeros of find_x_p_apr03_functional to find x_p in the nuclear matter phase
    def find_x_p_apr03(self,rho,x_p_guess,LDP_HDP):
        for i in range(10):
            x_p,info,status,message= optimize.fsolve(self.find_x_p_apr03_functional,x_p_guess,args=(rho,LDP_HDP),full_output = True)
            if(message!="The solution converged."):
                if(np.shape(x_p_guess)==()):
                    x_p_guess += random.randrange(-10,10)/100*x_p_guess
                else:
                    x_p_guess[0] += random.randrange(-10,10)/100*x_p_guess[0]
            else:
                return x_p
        if(abs(self.find_x_p_apr03_functional(x_p,rho,LDP_HDP))>self.tol):
            print("Solver find_x_p_apr03 failed to find x_p")
        return x_p 
    
    #Build the full low density (LDP) and high density (HDP) phase of APR03 for neutron matter 
    def build_full_LDP_HDP_apr03_EoS(self):
        self.rho_apr03_vec = np.linspace(self.rho_min_apr03,self.rho_max_apr03,self.N_apr03)
        self.x_p_LDP_apr03_vec = np.zeros(self.N_apr03)
        self.e_LDP_apr03_vec = np.zeros(self.N_apr03)
        self.mu_n_LDP_apr03_vec = np.zeros(self.N_apr03)
        self.mu_p_LDP_apr03_vec = np.zeros(self.N_apr03)
        self.x_p_LDP_guess = 0.01

        self.x_p_HDP_apr03_vec = np.zeros(self.N_apr03)
        self.e_HDP_apr03_vec = np.zeros(self.N_apr03)
        self.mu_n_HDP_apr03_vec = np.zeros(self.N_apr03)
        self.mu_p_HDP_apr03_vec = np.zeros(self.N_apr03)
        self.x_p_HDP_guess = 0.01

        for i in range(self.N_apr03):
            self.x_p_LDP_apr03_vec[i] = self.find_x_p_apr03(self.rho_apr03_vec[i],self.x_p_LDP_guess,0)
            
            self.e_LDP_apr03_vec[i] = self.total_energy_density_functional_apr03(self.rho_apr03_vec[i],self.x_p_LDP_apr03_vec[i],0)
            
            self.mu_n_LDP_apr03_vec[i],self.mu_p_LDP_apr03_vec[i] = (
                    self.chemical_potential_apr03_functional(self.rho_apr03_vec[i],self.x_p_LDP_apr03_vec[i],0))

            x_p_LDP_guess = self.x_p_LDP_apr03_vec[i]
             
            self.x_p_HDP_apr03_vec[i] = self.find_x_p_apr03(self.rho_apr03_vec[i],self.x_p_HDP_guess,1)
            self.e_HDP_apr03_vec[i] = self.total_energy_density_functional_apr03(self.rho_apr03_vec[i],self.x_p_HDP_apr03_vec[i],1)
            
            self.mu_n_HDP_apr03_vec[i],self.mu_p_HDP_apr03_vec[i] = (
                    self.chemical_potential_apr03_functional(self.rho_apr03_vec[i],self.x_p_HDP_apr03_vec[i],1))
            
            x_p_HDP_guess = self.x_p_HDP_apr03_vec[i]
       
        self.E_per_A_LDP_apr03_vec = self.e_LDP_apr03_vec/self.rho_apr03_vec-self.m
        self.E_per_A_HDP_apr03_vec = self.e_HDP_apr03_vec/self.rho_apr03_vec-self.m
        
        self.mu_e_LDP_apr03_vec = self.mu_n_LDP_apr03_vec-self.mu_p_LDP_apr03_vec
        self.pF_e_LDP_apr03_vec = np.heaviside(self.mu_e_LDP_apr03_vec-self.m_e,0)*np.sqrt(abs(self.mu_e_LDP_apr03_vec**2-self.m_e**2))
        self.rho_e_LDP_apr03_vec = self.pF_e_LDP_apr03_vec**3/(3*np.pi**2*self.hc**3)
        
        self.pF_mu_LDP_apr03_vec = np.heaviside(self.mu_e_LDP_apr03_vec-self.m_mu,0)*np.sqrt(abs(self.mu_e_LDP_apr03_vec**2-self.m_mu**2))
        self.rho_mu_LDP_apr03_vec = self.pF_mu_LDP_apr03_vec**3/(3*np.pi**2*self.hc**3)
            
        self.mu_e_HDP_apr03_vec = self.mu_n_HDP_apr03_vec-self.mu_p_HDP_apr03_vec
        self.pF_e_HDP_apr03_vec = np.heaviside(self.mu_e_HDP_apr03_vec-self.m_e,0)*np.sqrt(abs(self.mu_e_HDP_apr03_vec**2-self.m_e**2))
        self.rho_e_HDP_apr03_vec = self.pF_e_HDP_apr03_vec**3/(3*np.pi**2*self.hc**3)
        
        self.pF_mu_HDP_apr03_vec = np.heaviside(self.mu_e_HDP_apr03_vec-self.m_mu,0)*np.sqrt(abs(self.mu_e_HDP_apr03_vec**2-self.m_mu**2))
        self.rho_mu_HDP_apr03_vec = self.pF_mu_HDP_apr03_vec**3/(3*np.pi**2*self.hc**3)
       
        self.E_per_A_LDP_with_lepton_vec = (self.E_per_A_LDP_apr03_vec
                                           +self.mu_e_LDP_apr03_vec*(self.rho_e_LDP_apr03_vec+self.rho_mu_LDP_apr03_vec))
        
        self.E_per_A_HDP_with_lepton_vec = (self.E_per_A_HDP_apr03_vec
                                           +self.mu_e_HDP_apr03_vec*(self.rho_e_HDP_apr03_vec+self.rho_mu_HDP_apr03_vec))



        self.P_LDP_apr03_vec = (self.mu_n_LDP_apr03_vec*self.rho_apr03_vec*(1-self.x_p_LDP_apr03_vec)
                               +self.mu_p_LDP_apr03_vec*self.rho_apr03_vec*self.x_p_LDP_apr03_vec
                               +self.mu_e_LDP_apr03_vec*self.rho_e_LDP_apr03_vec
                               +self.mu_e_LDP_apr03_vec*self.rho_mu_LDP_apr03_vec
                               -self.e_LDP_apr03_vec)
        
        self.P_HDP_apr03_vec = (self.mu_n_HDP_apr03_vec*self.rho_apr03_vec*(1-self.x_p_HDP_apr03_vec)
                               +self.mu_p_HDP_apr03_vec*self.rho_apr03_vec*self.x_p_HDP_apr03_vec
                               +self.mu_e_HDP_apr03_vec*self.rho_e_HDP_apr03_vec
                               +self.mu_e_HDP_apr03_vec*self.rho_mu_HDP_apr03_vec
                               -self.e_HDP_apr03_vec)


        return

    #Functional used to find the density at which the transition from LDP to HDP starts
    #This is the density the energy per nucleon is equal in both LDP and HDP
    def LDP_HDP_apr03_intersect_rho_functional(self,rho):
        return (np.interp(rho,self.rho_apr03_vec,self.E_per_A_LDP_with_lepton_vec)
                -np.interp(rho,self.rho_apr03_vec,self.E_per_A_HDP_with_lepton_vec))
    
    #Functional used to find the density at which the transition from LDP to HDP ends
    #This is where the pressure in HDP is equal to that of LDP in the beginning of 
    #the transition
    def LDP_HDP_apr03_intersect_P_functional(self,rho):
        return (np.interp(rho,self.rho_apr03_vec,self.P_HDP_apr03_vec)
                -np.interp(self.rho_LDP_apr03_intersect,self.rho_apr03_vec,self.P_LDP_apr03_vec))

    #Function that gives the two densities at which the transition between LDP and HDP happens
    def LDP_HDP_transition(self):
        self.rho_LDP_transition_index = 0
        for i in range(self.N_apr03):
            if(self.E_per_A_LDP_with_lepton_vec[i]>self.E_per_A_HDP_with_lepton_vec[i]):
                self.rho_LDP_apr03_intersect = optimize.fsolve(
                        self.LDP_HDP_apr03_intersect_rho_functional,[self.rho_apr03_vec[i]])
                self.rho_LDP_transition_index = i
                break

        if(abs(self.LDP_HDP_apr03_intersect_rho_functional(self.rho_LDP_apr03_intersect)>self.tol)):
            print("LDP_HDP_transition failed to find initial density")
        
        for i in range(self.rho_LDP_transition_index,self.N_apr03):
            if(self.P_HDP_apr03_vec[i]>self.P_LDP_apr03_vec[self.rho_LDP_transition_index]):
                rho_guess = [self.rho_apr03_vec[i]+0.2]
                for j in range(10):
                    rho,info,status,message= optimize.fsolve(
                        self.LDP_HDP_apr03_intersect_P_functional,rho_guess,full_output = True)
                    if(status!=1):
                        rho_guess = [rho_guess[0]+random.randrange(-10,10)/100]
                    else:
                        break
                self.rho_HDP_apr03_intersect = rho 
                self.rho_HDP_transition_index = i
                break

        if(abs(self.LDP_HDP_apr03_intersect_P_functional(self.rho_HDP_apr03_intersect))>self.tol):
            print("LDP_HDP_transition failed to find end density")

    #Using the transition densities and the full APR LDP and HDP, compute the connected APR03 EoS
    def build_connected_apr03_EoS(self):
        self.N_apr03_combined = self.N_apr03-(self.rho_HDP_transition_index-self.rho_LDP_transition_index)+2
        self.x_p_apr03_combined_vec = np.zeros(self.N_apr03_combined)
        self.rho_apr03_combined_vec = np.zeros(self.N_apr03_combined)
        self.e_apr03_combined_vec = np.zeros(self.N_apr03_combined)
        self.mu_n_apr03_combined_vec = np.zeros(self.N_apr03_combined)
        self.mu_p_apr03_combined_vec = np.zeros(self.N_apr03_combined)

        
        #Everything from here to ##### is connecting the EoS LDP to HDP at the transtition
        self.rho_apr03_combined_vec[0:self.rho_LDP_transition_index] = self.rho_apr03_vec[0:self.rho_LDP_transition_index]
        self.rho_apr03_combined_vec[self.rho_LDP_transition_index+2:] = self.rho_apr03_vec[self.rho_HDP_transition_index:]
        self.rho_apr03_combined_vec[self.rho_LDP_transition_index] = self.rho_LDP_apr03_intersect
        self.rho_apr03_combined_vec[self.rho_LDP_transition_index+1] = self.rho_HDP_apr03_intersect
        
        
        self.x_p_apr03_combined_vec[0:self.rho_LDP_transition_index] = self.x_p_LDP_apr03_vec[0:self.rho_LDP_transition_index]
    
        self.x_p_apr03_combined_vec[self.rho_LDP_transition_index+2:] = self.x_p_HDP_apr03_vec[self.rho_HDP_transition_index:]
    
        self.x_p_apr03_combined_vec[self.rho_LDP_transition_index] = (
                self.find_x_p_apr03(self.rho_LDP_apr03_intersect,self.x_p_LDP_apr03_vec[self.rho_LDP_transition_index],0))
        
        self.x_p_apr03_combined_vec[self.rho_LDP_transition_index+1] = (
                self.find_x_p_apr03(self.rho_HDP_apr03_intersect,self.x_p_HDP_apr03_vec[self.rho_HDP_transition_index],1))
        
        self.e_apr03_combined_vec[0:self.rho_LDP_transition_index] = self.e_LDP_apr03_vec[0:self.rho_LDP_transition_index]
        self.e_apr03_combined_vec[self.rho_LDP_transition_index+2:] = self.e_HDP_apr03_vec[self.rho_HDP_transition_index:]
        self.e_apr03_combined_vec[self.rho_LDP_transition_index] = (
                self.total_energy_density_functional_apr03(self.rho_LDP_apr03_intersect,self.x_p_apr03_combined_vec[self.rho_LDP_transition_index],0))
        self.e_apr03_combined_vec[self.rho_LDP_transition_index+1] = (
                self.total_energy_density_functional_apr03(self.rho_HDP_apr03_intersect,self.x_p_apr03_combined_vec[self.rho_LDP_transition_index+1],1))

        self.mu_n_apr03_combined_vec[0:self.rho_LDP_transition_index] = self.mu_n_LDP_apr03_vec[0:self.rho_LDP_transition_index]
        self.mu_n_apr03_combined_vec[self.rho_LDP_transition_index+2:] = self.mu_n_HDP_apr03_vec[self.rho_HDP_transition_index:]
        
        self.mu_n_apr03_combined_vec[self.rho_LDP_transition_index],self.mu_p_apr03_combined_vec[self.rho_LDP_transition_index] = (
                self.chemical_potential_apr03_functional(self.rho_LDP_apr03_intersect,self.x_p_apr03_combined_vec[self.rho_LDP_transition_index],0))
        self.mu_n_apr03_combined_vec[self.rho_LDP_transition_index+1],self.mu_p_apr03_combined_vec[self.rho_LDP_transition_index+1] = (
                self.chemical_potential_apr03_functional(self.rho_HDP_apr03_intersect,self.x_p_apr03_combined_vec[self.rho_LDP_transition_index+1],1))

        self.mu_p_apr03_combined_vec[0:self.rho_LDP_transition_index] = self.mu_p_LDP_apr03_vec[0:self.rho_LDP_transition_index]
        self.mu_p_apr03_combined_vec[self.rho_LDP_transition_index+2:] = self.mu_p_HDP_apr03_vec[self.rho_HDP_transition_index:]
        #####
        

        self.mu_e_apr03_combined_vec = self.mu_n_apr03_combined_vec-self.mu_p_apr03_combined_vec
        self.pF_e_apr03_combined_vec = np.heaviside(self.mu_e_apr03_combined_vec-self.m_e,0)*np.sqrt(self.mu_e_apr03_combined_vec**2-self.m_e**2)
        self.rho_e_apr03_combined_vec = self.pF_e_apr03_combined_vec**3/(3*np.pi**2*self.hc**3)
    
        self.pF_mu_apr03_combined_vec = np.heaviside(self.mu_e_apr03_combined_vec-self.m_mu,0)*np.sqrt(abs(self.mu_e_apr03_combined_vec**2-self.m_mu**2))
        self.rho_mu_apr03_combined_vec = self.pF_mu_apr03_combined_vec**3/(3*np.pi**2*self.hc**3)

        self.P_apr03_combined_vec = (self.mu_n_apr03_combined_vec*self.rho_apr03_combined_vec*(1-self.x_p_apr03_combined_vec)
                                        +self.mu_p_apr03_combined_vec*self.rho_apr03_combined_vec*self.x_p_apr03_combined_vec
                                        +self.mu_e_apr03_combined_vec*self.rho_e_apr03_combined_vec
                                        +self.mu_e_apr03_combined_vec*self.rho_mu_apr03_combined_vec
                                        -self.e_apr03_combined_vec)

        

    
       



#Class for computing the color flavored locked (CFL) phase (or just the exotic phase)
class CFL_EoS(object):
    def __init__(self,N_CFL                     #Number of points used to compute CFL phase
                     ,N_CFL_kaons               #Number of points used to compute mixed kaon and CFL phase
                     ,B                         #Bag constant
                     ,Delta                     #Pairing gap
                     ,m_s                       #Strange quark mass
                     ,mu_q_min = 319.422        #Minimum quark chemical potential we start computing CFL from 
                     ,mu_q_max = 700            #Maximum quark chemical potential we compute the CFL phase to
                     ,N_apr03 = 400             #Number of points used in APR EoS
                     ,rho_min_apr03 = 0.1       #Lowest number density we compute APR03 from 
                     ,rho_max_apr03 = 1         #Max density we compute APR03 to
                     ,d_rho = 0.0001            #When computing derivatives in APR, we use this step size in rho
                     ,tol = 1e-6                #Absolute tolerance used in root solvers finding electron density and proton fraction  
                     ,c = 0.                    #Phenomenological parameter for quark interactions. Usually set to 0.3
                     ,eos_apr03 = None          #If we already have computed the EoS for APR03, we can insert it here, so we don't have to
                                                #compute it again 
                     ,TOV=False                 #If set to True, compute MR curves in addition to EoS
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
        self.build_CFL_EoS()
        
        
        self.N_CFL_kaons = N_CFL_kaons
        self.N_apr03 = N_apr03
        self.rho_min_apr03 = rho_min_apr03
        self.rho_max_apr03 = rho_max_apr03
        self.d_rho = d_rho
        
        if(eos_apr03==None):
            self.eos_apr03 = apr03_EoS(self.N_apr03
                             ,rho_min_apr03 = self.rho_min_apr03
                             ,rho_max_apr03 = self.rho_max_apr03
                             ,d_rho = self.d_rho)

        else:
            self.eos_apr03 = eos_apr03

        self.build_CFL_with_kaons_EoS()

        self.full_eos()

        if(TOV==True):
            self.v2_vec = np.gradient(self.P_vec,self.e_vec,edge_order=2)
            R_vec,M_vec,Lambda_vec,P_c_vec = TOV_Rahul.tov_solve(self.e_vec,self.P_vec,self.v2_vec)
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
        if(self.kaon_mass_squared(mu_q)<mu_e**2):
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
                    -self.eos_apr03.lepton_energy_density_apr03(mu_e,self.m_e)
                    -self.eos_apr03.lepton_energy_density_apr03(mu_e,self.m_mu))
        return self.pressure_CFL(mu_q,pF)
    
    #Total energy density of CFL when kaons are included
    def energy_density_CFL_with_kaons(self,mu_q,mu_e):
        if(self.kaon_mass_squared(mu_q)>mu_e**2):
            return 0
        return (self.f_pi_squared(mu_q)*mu_e**2*(
                1+2*self.kaon_mass_squared(mu_q)/mu_e**2-3*self.kaon_mass_squared(mu_q)**2/mu_e**4)/(2*self.hc**3)
                +self.eos_apr03.lepton_energy_density_apr03(mu_e,self.m_e)
                +self.eos_apr03.lepton_energy_density_apr03(mu_e,self.m_mu)) 
    
    #Kaon number density
    def kaon_density_CFL(self,mu_q,mu_e):
        if(self.kaon_mass_squared(mu_q)>mu_e**2):
            return 0
        return self.f_pi_squared(mu_q)*mu_e*(1-self.kaon_mass_squared(mu_q)**2/mu_e**4)/(self.hc**3)

    
    #Function that is (0,0,0) when we have found a density, proton fraction and electron chemical potential are 
    #in equilibrium with kaon condensate
    def chemical_potential_electron_CFL_with_kaons_functional(self,mu_e_x_p_rho,mu_q,pF):
        mu_e,x_p,rho = mu_e_x_p_rho
        mu_n = 3*mu_q
        mu_p = mu_n-mu_e
        mu_n_calc,mu_p_calc = self.eos_apr03.chemical_potential_apr03_functional(rho,x_p,1)
        f1 = ((self.eos_apr03.total_pressure_of_mu_e_apr03_test(mu_n,mu_e,x_p,rho,1)
                -self.pressure_CFL_with_kaons(mu_q,pF,mu_e)))
        f2 = mu_n-mu_n_calc
        f3 = mu_p-mu_p_calc
        return f1,f2,f3

    #Find the zeros of chemical_potential_electron_CFL_kaons_functional in order to 
    #determine electron chemical potential, proton fraction, and total density 
    def chemical_potential_electron_CFL_with_kaons(self,mu_q,pF,mu_e_x_p_rho_guess):
        mu_e_x_p_rho,info,status,message=(
        optimize.fsolve(self.chemical_potential_electron_CFL_with_kaons_functional,mu_e_x_p_rho_guess,args=(mu_q,pF),full_output = True))
        return mu_e_x_p_rho,status


    #Build EoS with mixed phase of CFL and kaons
    def build_CFL_with_kaons_EoS(self):
        self.mu_e_CFL_with_kaons_vec = np.zeros(self.N_CFL_kaons)
        self.mu_e_CFL_with_kaons_vec_test = np.zeros(self.N_CFL_kaons)
        self.P_CFL_with_kaons_vec = np.zeros(self.N_CFL_kaons)
        self.mu_q_CFL_with_kaons_vec = np.linspace(self.mu_q_min,self.mu_q_max,self.N_CFL_kaons)
        self.e_mixed_phase_vec = np.zeros(self.N_CFL_kaons)
        self.rho_mixed_phase_vec = np.zeros(self.N_CFL_kaons)
        status = 1
        end = False
        pF_guess = 0.1
        self.mix_index_start = 0
        self.mix_index_end = 0
        for i in range(self.N_CFL_kaons):
            if(self.P_CFL_vec[0]>self.eos_apr03.P_apr03_combined_vec[0]):
                self.status = "Fail"
                return 
            mu_q = self.mu_q_CFL_with_kaons_vec[i]
            pF = self.fermi_momenta_CFL(mu_q,pF_guess)[0]
            pF_guess = pF
            if(i == 0 or status != 1):
                mu_e_guess = np.interp(3*mu_q,self.eos_apr03.mu_n_apr03_combined_vec,self.eos_apr03.mu_e_apr03_combined_vec)
                x_p_guess = np.interp(3*mu_q,self.eos_apr03.mu_n_apr03_combined_vec,self.eos_apr03.x_p_apr03_combined_vec)
                rho_guess = np.interp(3*mu_q,self.eos_apr03.mu_n_apr03_combined_vec,self.eos_apr03.rho_apr03_combined_vec)
                mu_e_x_p_rho_guess = [mu_e_guess,x_p_guess,rho_guess]
        
            if(self.mix_index_start == 0 and status == 1):
                    self.mix_index_start = i


            if(end!=True): 
                mu_e_x_p_rho_NM,status = self.chemical_potential_electron_CFL_with_kaons(mu_q,pF,mu_e_x_p_rho_guess)
                mu_e,x_p,rho_NM = mu_e_x_p_rho_NM
            else:
                mu_e = 0
                x_p = 0
                rho_NM = 0
            

            if(status != 1):
                mu_e = 0
                x_p = 0
                rho_NM = 0

            if(status == 1 and mu_e<0):
                mu_e=0
                x_p = 0
                end = True
                self.mix_index_end = i

            mu_e_x_p_rho_guess = [mu_e,x_p,rho_NM]
            self.P_CFL_with_kaons_vec[i] = self.pressure_CFL_with_kaons(mu_q,pF,mu_e)
            self.mu_e_CFL_with_kaons_vec[i] = mu_e

            rho_CFL_with_kaons = self.density_CFL(mu_q,pF)

            e_CFL_kaons = (self.energy_density_CFL(mu_q,pF)+self.energy_density_CFL_with_kaons(mu_q,mu_e))
                   

            e_NM = (self.eos_apr03.apr03_functional(rho_NM,x_p)[1]
                    +self.eos_apr03.lepton_energy_density_apr03(mu_e,self.m_e)
                    +self.eos_apr03.lepton_energy_density_apr03(mu_e,self.m_mu))

            
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
            


            if(mu_e==0):
                if(self.mix_index_end>0):
                    chi = 1
                else:
                    chi = 0
            else:
                chi = (x_p*rho_NM-rho_e-rho_mu)/(x_p*rho_NM+self.kaon_density_CFL(mu_q,mu_e))


            rho_CFL_with_kaons+=rho_e+rho_mu 
            self.e_mixed_phase_vec[i] = (1-chi)*e_NM + chi*e_CFL_kaons
            self.rho_mixed_phase_vec[i] = rho_NM*(1-chi)+chi*rho_CFL_with_kaons
                      

        return

    #Function that returns zero when electron chemical potential is equal in neutron matter and the mixed phase
    def find_transition_mu_q_to_mix_phase_functional(self,mu_q):
        mu_q_NM_vec = self.eos_apr03.mu_n_HDP_apr03_vec/3
        mu_e_NM_vec = self.eos_apr03.mu_e_HDP_apr03_vec
        mu_q_CFL_vec = self.mu_q_CFL_with_kaons_vec
        mu_e_mix_vec = self.mu_e_CFL_with_kaons_vec
        return np.interp(mu_q,mu_q_CFL_vec,mu_e_mix_vec)-np.interp(mu_q,mu_q_NM_vec,mu_e_NM_vec)
    
    #Finds the transition density from neutron matter to mixed phase by solving for zeros of find_transition_mu_q_to_mix_phase_functional
    def find_transition_mu_q_to_mix_phase(self,mu_q_guess):
        mu_q_guess_temp = mu_q_guess
        for i in range(10):
            mu_q_intersect,info,status,message = optimize.fsolve(self.find_transition_mu_q_to_mix_phase_functional,mu_q_guess,full_output =
                    True)
            if(status!=1):
                if(np.shape(mu_q_guess == ())):
                    mu_q_guess = mu_q_guess+mu_q_guess_temp*random.randrange(-10,10)/100
                else:
                    mu_q_guess = [mu_q_guess_temp[0]+mu_q_guess_temp[0]*random.randrange(-10,10)/100]
            else:
                break
        if(i==9):
            self.status = "Fail"
            print('find_transition_mu_q_to_mix_phase failed')
        return mu_q_intersect[0] 


    #Compute full equation of state with APR03 at low density, kaon with CFL mixed phase, and pure CFL 
    def full_eos(self):
    
        NM = True
        MIX = False
        CFL = False
        self.N = self.eos_apr03.N_apr03_combined+self.N_CFL+self.N_CFL_kaons
        self.mu_q_vec = np.zeros(self.N)
        self.P_vec = np.zeros(self.N)
        self.e_vec = np.zeros(self.N)
        self.rho_vec = np.zeros(self.N)
        
        
        if(self.status == "Fail"):
            self.P_vec = self.eos_apr03.P_apr03_combined_vec
            self.e_vec = self.eos_apr03.e_apr03_combined_vec
            self.rho_vec = self.eos_apr03.rho_apr03_combined_vec
            self.mu_q_vec = self.eos_apr03.mu_n_apr03_combined_vec/3
            return 
        i_mix = 0
        i_CFL = 0
        end = False
        self.mu_q_transition = self.find_transition_mu_q_to_mix_phase([self.mu_q_CFL_with_kaons_vec[int((self.mix_index_start+self.mix_index_end)/2)]])
        add_extra = True
        for i in range(self.N): 
            if(NM==True):
                mu_q = self.eos_apr03.mu_n_apr03_combined_vec[i]/3
                self.mu_q_vec[i] = mu_q
                P_NM = self.eos_apr03.P_apr03_combined_vec[i]
                mu_e_NM = self.eos_apr03.mu_e_apr03_combined_vec[i]
                if(mu_q <= self.mu_q_transition):
                    self.P_vec[i] = P_NM
                    self.mu_q_vec[i] = mu_q
                    self.rho_vec[i] = np.interp(P_NM,self.eos_apr03.P_apr03_combined_vec,self.eos_apr03.rho_apr03_combined_vec)
                    self.e_vec[i] = np.interp(P_NM,self.eos_apr03.P_apr03_combined_vec,self.eos_apr03.e_apr03_combined_vec)
                else:
                    while(self.mu_q_CFL_with_kaons_vec[self.mix_index_start+i_mix]<self.mu_q_transition):
                        i_mix+=1
                        if(self.mix_index_start+i_mix>=self.N_CFL_kaons):
                            end=True
                            break
                    while(self.e_mixed_phase_vec[self.mix_index_start+i_mix]>self.e_mixed_phase_vec[self.mix_index_start+i_mix+1]):
                        i_mix+=1
                        print(i_mix)
                        if(self.mix_index_start+i_mix>=self.N_CFL_kaons):
                            end=True
                            break

                    if(end==True):
                        break
                    MIX = True
                    NM = False
                    self.mu_q_vec[i] = self.mu_q_CFL_with_kaons_vec[self.mix_index_start+i_mix]
                    P_mix = self.P_CFL_with_kaons_vec[self.mix_index_start+i_mix]
                    self.P_vec[i] = P_mix
                    self.rho_vec[i] = self.rho_mixed_phase_vec[self.mix_index_start+i_mix]
                    self.e_vec[i] = self.e_mixed_phase_vec[self.mix_index_start+i_mix]
                    i_mix+=1
            elif(MIX == True):
                    self.mu_q_vec[i] = self.mu_q_CFL_with_kaons_vec[self.mix_index_start+i_mix]
                    P_mix = self.P_CFL_with_kaons_vec[self.mix_index_start+i_mix]
                    self.P_vec[i] = P_mix
                    self.rho_vec[i] = self.rho_mixed_phase_vec[self.mix_index_start+i_mix]
                    self.e_vec[i] = self.e_mixed_phase_vec[self.mix_index_start+i_mix]
                    i_mix+=1


            if(self.mix_index_start+i_mix>=self.N_CFL_kaons):
                break
        
        #Remove zeros from P, e and rho vecs
        new_P_vec = []
        new_e_vec = []
        new_rho_vec = []

        for i in range(len(self.P_vec)):
            if(self.P_vec[i]!=0):
                new_P_vec.append(self.P_vec[i])
                new_e_vec.append(self.e_vec[i])
                new_rho_vec.append(self.rho_vec[i])
        self.P_vec = np.array(new_P_vec)
        self.e_vec = np.array(new_e_vec)
        self.rho_vec = np.array(new_rho_vec)


        
            


