def energy_density_CFL_1(self,mu_q,pF):
    return (3*(1-self.c)*self.energy_density_free_quark_CFL(pF,self.m_u)
            +3*(1-self.c)*self.energy_density_free_quark_CFL(pF,self.m_d)
            +3*self.energy_density_free_quark_CFL(pF,self.m_s)
            -3*self.c*self.energy_density_free_quark_CFL(pF,0)
            -(9*self.c*mu_q*pF**2/np.pi**2)*(2-mu_q/np.sqrt(mu_q**2+self.m_s**2/3))*(mu_q-pF)
            +3*mu_q**2*self.Delta**2/np.pi**2+self.B**4)/self.hc**3

def energy_density_CFL_2(self,mu_q,pF):
    return (3*(1-self.c)*self.energy_density_free_quark_CFL(pF,self.m_u)
               +3*(1-self.c)*self.energy_density_free_quark_CFL(pF,self.m_d)
               +3*self.energy_density_free_quark_CFL(pF,self.m_s)
               -3*self.c*self.energy_density_free_quark_CFL(pF,0)
               -(9*self.c*mu_q*pF**2/np.pi**2)*(2-mu_q/np.sqrt(mu_q**2+self.m_s**2/3))*(mu_q-pF)
               +3*mu_q**2*self.Delta**2/np.pi**2+self.B**4)/self.hc**3


a,b = [1,2]
print(energy_density_CFL_1(a,b))
