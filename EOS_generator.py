import numpy as np
from scipy.optimize import root
import astropy.constants as c
import warnings
warnings.filterwarnings('ignore') # :^)




class EOS_set:
    def __init__(self
                ,N_RMF=100
                 ,FSUGold_filename = "FSUGoldgarnet.inp"):#"FSUGold.inp"):

        self.N_RMF = N_RMF
        self.FSUGold_filename = FSUGold_filename
        self.constants()
        self.rho_C = self.get_rhoC()
        self.e_vec,self.P_vec,self.rho_vec,self.mu_e_vec,self.mu_n_vec,self.x_p_vec,self.fields_vec = self.get_core_EOS(self.rho_C)
        return

    def constants(self):
        self.ms,self.gs2,self.gv2,self.gp2,self.gd2,self.kappa,self.lambd,self.zeta,self.lambda_v = np.loadtxt(self.FSUGold_filename)

        #these are always going to be zero
        self.lambda_s = 0
        self.lambda_d = 0
        self.lambda_1,self.lambda_2,self.lambda_3=0,0,0
        self.xi = 0


        if(self.FSUGold_filename=="FSUGold_MARK.inp"):
            self.m_nuc=938.91 #MARK
            self.mv = 783. #MARK
            self.mp = 769. #MARK
            self.m_m = 105.66 #MARK
            self.m = 939.565 #MARK
            self.m_e = 0.00001 #MARK

        else:
            self.m = 939 #mass of nucleon
            self.m_nuc = self.m
            self.m_m = 105.7 #muon mass
            self.m_e = .511 #electron mass
            self.mv = 782.5 #vector meson mass
            self.mp = 763 #rho meson mass

        self.md = 980 #delta meson mass
        self.hbc = (c.hbar*c.c).to('MeV fm').value
        self.hbc3 = self.hbc**3

    def get_kF(self,dens):
        if(dens>0):
            return (3*np.pi**2*dens)**(1/3)
        return 0
    def sca_dens(self,kF,s,d,i):
        """Returns the scalar density"""
        mdirac = self.m-s+i*d/2
        return mdirac*(kF*np.sqrt(kF**2+mdirac**2)-mdirac**2*np.arcsinh(kF/mdirac))/2./np.pi**2
    def get_ms_eff(self,Phi,W0,B0):
        return np.sqrt(self.ms**2+self.gs2*(self.kappa*Phi+self.lambd/2*Phi**2-2.*(self.lambda_2*W0**2+self.lambda_s*B0**2)))
    def get_mv_eff(self,Phi,W0,B0):
        return np.sqrt(self.mv**2+self.gv2*(self.zeta/2*W0**2+2*self.lambda_v*B0**2+2*(Phi*self.lambda_1+self.lambda_2*Phi**2)))
    def get_mr_eff(self,Phi,W0,B0):
        return np.sqrt(self.mp**2+self.gp2*2*(self.lambda_v*W0**2+self.lambda_3*Phi+self.lambda_s*Phi**2)+self.gp2*self.xi/2*B0**2)
    #Everything from here
    def pi_s(self,q,rho,m_eff):
        kF = self.get_kF(rho)
        EF = np.sqrt(kF**2+m_eff**2)
        E = np.sqrt(q**2/4+m_eff**2)
        T1 = (kF*EF-(3*m_eff**2+q**2/2)*np.log((kF+EF)/m_eff)+2*EF*E**2/q*np.log(abs((2*kF-q)/(2*kF+q))))/2/np.pi**2
        T2 = (2*E**3/q*np.log(abs((q*EF-2*kF*E)/(q*EF+2*kF*E))))/2/np.pi**2
        return T1-T2
    def pi_m(self,q,rho,m_eff):
        kF = self.get_kF(rho)
        return m_eff/2/np.pi**2*(kF-(kF**2/q-q/4)*np.log(abs((2*kF-q)/(2*kF+q))))
    def pi_00(self,q,rho,m_eff):
        kF = self.get_kF(rho)
        EF = np.sqrt(kF**2+m_eff**2)
        E = np.sqrt(q**2/4+m_eff**2)
        T1 = -1/np.pi**2*(2/3*kF*EF-q**2/6*np.log((kF+EF)/m_eff)-EF/3/q*(EF**2-3/4*q**2)*np.log(abs((2*kF-q)/(2*kF+q))))
        T2 = 1/np.pi**2*(E/3/q*(m_eff**2-q**2/2)*np.log(abs((q*EF-2*kF*E)/(q*EF+2*kF*E))))
        return T1-T2
    def delta_pol_mat(self,q,rho,Yp,Ye,sca_field,d_field):
        Yn = 1-Yp
        mps = self.m_nuc-sca_field-.5*d_field
        mns = mps + d_field
        R1 = [self.pi_00(q,rho*Ye,self.m_e),0,0,0,0]
        R2 = [0,self.pi_s(q,rho*Yp,mps),self.pi_m(q,Yp*rho,mps),0,0]
        R3 = [0,self.pi_m(q,Yp*rho,mps),self.pi_00(q,Yp*rho,mps),0,0]
        R4 = [0,0,0,self.pi_s(q,rho*Yn,mns),self.pi_m(q,Yn*rho,mns)]
        R5 = [0,0,0,self.pi_m(q,Yn*rho,mns),self.pi_00(q,rho*Yn,mns)]
        return np.matrix([R1,R2,R3,R4,R5])

    def delt_prop_mat(self,q,dens,Yp,Phi,W0,B0,D0):
        Fs = q**2+self.get_ms_eff(Phi, W0, B0)**2
        Fv = q**2+self.get_mv_eff(Phi, W0, B0)**2
        Fp = q**2+self.get_mr_eff(Phi, W0, B0)**2
        Lsv2 = 4*self.gs2*self.gv2*W0**2*(self.lambda_1+2*self.lambda_2*Phi)**2
        Lsp2 = 4*self.gs2*self.gp2*B0**2*(self.lambda_3+2*self.lambda_s*Phi)**2
        Lvp2 = 16*self.gp2*self.gv2*W0**2*B0**2*self.lambda_v**2
        H = Fs*Fv*Fp+Lsv2*Fp+Lsp2*Fv-Lvp2*Fs

        dg = 4*np.pi*c.alpha/q**2
        ds = self.gs2*Fv*Fp/(Fs*Fv*Fp+Lsv2*Fp+Lsp2*Fv)
        dv = self.gv2*Fs*Fp/(Fs*Fv*Fp+Lsv2*Fp-Lvp2*Fs)
        dr = self.gp2*Fv*Fs/4/(Fs*Fv*Fp+Lsp2*Fv-Lvp2*Fs)
        dd = self.gd2/4/(q**2+self.md**2)

        dsv = Fp*np.sqrt(self.gs2*self.gv2*Lsv2)/H
        dsr = Fv*np.sqrt(self.gs2*self.gp2*Lsp2)/H/2*np.sign(B0)
        dvr = -Fs*np.sqrt(self.gp2*self.gv2*Lvp2)/H/2*np.sign(B0)

        R1 = [dg,   0,             -dg,           0,                     0]
        R2 = [0,    -ds-dd,        -dsv+dsr,      -ds+dd,         -dsv-dsr]
        R3 = [-dg,  -dsv+dsr,      dg+dv+dr+2*dvr, -dsv+dsr,         dv-dr]
        R4 = [0,    -ds+dd,        -dsv+dsr,      -ds-dd,         -dsv-dsr]
        R5 = [0,    -dsv-dsr,      dv-dr,         -dsv-dsr,    dv+dr-2*dvr]
        return np.matrix([R1,R2,R3,R4,R5])

    def delt_dielectric(self,q,dens,Yp,Ye,sF,vF,rF,dF):
        DL = self.delt_prop_mat(q, dens,Yp,sF,vF,rF,dF)
        Pi_L = self.delta_pol_mat(q,dens,Yp,Ye,sF,dF)
        return np.linalg.det(np.identity(5)-np.matmul(DL,Pi_L))
    #to here is for the core-crust transition density

    def solve_mesons(self,dens,Yp,x0=[300,200,-10,-10]):
        """
        Solves for the nonlinear meson fields given input density and proton fraction.
        Returns [S,V,B,D]
        """
        kFp = self.get_kF(dens*Yp)
        kFn = self.get_kF(dens*(1-Yp))

        nls = lambda F: (self.lambda_1+2*self.lambda_2*F[0])*F[1]**2+(self.lambda_3+2*self.lambda_s*F[0])*F[2]**2-self.kappa/2*F[0]**2-self.lambd/6*F[0]**3-2*self.lambda_d*F[0]*F[3]**2
        nlv = lambda F: (self.lambda_1*F[0]+self.lambda_2*F[0]**2)*2*F[1]
        nlp = lambda F: (self.lambda_3*F[0]+self.lambda_s*F[0]**2)*2*F[2]

        fields = lambda F: np.r_[F[0] - self.gs2/self.ms**2*(self.sca_dens(kFp,F[0],F[3],-1)+self.sca_dens(kFn,F[0],F[3],1)+nls(F)),
                            F[1] - self.gv2/self.mv**2*(dens-self.zeta/6*F[1]**3-2*self.lambda_v*F[1]*F[2]**2-nlv(F)),
                            F[2] - self.gp2/self.mp**2*((2*Yp-1)*dens/2-2*F[2]*(self.lambda_v*F[1]**2)-nlp(F)),
                            F[3] - self.gd2/self.md**2*((self.sca_dens(kFp,F[0],F[3],-1)-self.sca_dens(kFn,F[0],F[3],1))/2-2*self.lambda_d*F[3]*F[0]**2)]
        solution = root(fields,x0,method='lm')
        return solution.x


    def solve_mesons_kaons(self,dens,Yp,kFn,kFp,x0=[300,200,-10,-10]):
        """
        Solves for the nonlinear meson fields given input density and proton fraction.
        Returns [S,V,B,D]
        """
        #kFp = self.get_kF(dens*Yp)
        #kFn = self.get_kF(dens*(1-Yp))

        nls = lambda F: (self.lambda_1+2*self.lambda_2*F[0])*F[1]**2+(self.lambda_3+2*self.lambda_s*F[0])*F[2]**2-self.kappa/2*F[0]**2-self.lambd/6*F[0]**3-2*self.lambda_d*F[0]*F[3]**2
        nlv = lambda F: (self.lambda_1*F[0]+self.lambda_2*F[0]**2)*2*F[1]
        nlp = lambda F: (self.lambda_3*F[0]+self.lambda_s*F[0]**2)*2*F[2]

        fields = lambda F: np.r_[F[0] - self.gs2/self.ms**2*(self.sca_dens(kFp,F[0],F[3],-1)+self.sca_dens(kFn,F[0],F[3],1)+nls(F)),
                            F[1] - self.gv2/self.mv**2*(dens-self.zeta/6*F[1]**3-2*self.lambda_v*F[1]*F[2]**2-nlv(F)),
                            F[2] - self.gp2/self.mp**2*((2*Yp-1)*dens/2-2*F[2]*(self.lambda_v*F[1]**2)-nlp(F)),
                            F[3] - self.gd2/self.md**2*((self.sca_dens(kFp,F[0],F[3],-1)-self.sca_dens(kFn,F[0],F[3],1))/2-2*self.lambda_d*F[3]*F[0]**2)]
        solution = root(fields,x0,method='lm')
        return solution.x

    def mesons_kaons_functional(self,dens,Yp,kFn,kFp,fields):
        """
        Solves for the nonlinear meson fields given input density and proton fraction.
        Returns [S,V,B,D]
        """
        F = fields
        nls = (self.lambda_1+2*self.lambda_2*F[0])*F[1]**2+(self.lambda_3+2*self.lambda_s*F[0])*F[2]**2-self.kappa/2*F[0]**2-self.lambd/6*F[0]**3-2*self.lambda_d*F[0]*F[3]**2
        nlv = (self.lambda_1*F[0]+self.lambda_2*F[0]**2)*2*F[1]
        nlp = (self.lambda_3*F[0]+self.lambda_s*F[0]**2)*2*F[2]

        f1 =  F[0] - self.gs2/self.ms**2*(self.sca_dens(kFp,F[0],F[3],-1)+self.sca_dens(kFn,F[0],F[3],1)+nls)
        f2 =  F[1] - self.gv2/self.mv**2*(dens-self.zeta/6*F[1]**3-2*self.lambda_v*F[1]*F[2]**2-nlv)
        f3 = F[2] - self.gp2/self.mp**2*((2*Yp-1)*dens/2-2*F[2]*(self.lambda_v*F[1]**2)-nlp)
        f4 = F[3] - self.gd2/self.md**2*((self.sca_dens(kFp,F[0],F[3],-1)-self.sca_dens(kFn,F[0],F[3],1))/2-2*self.lambda_d*F[3]*F[0]**2)
        return f1,f2,f3,f4

    def muon_equil(self,kp):
        min_func = lambda ke: ke**3+(ke**2+self.m_e**2-self.m_m**2)**1.5-kp**3
        sol = root(min_func,kp)
        return sol.x[0]

    def beta_equil(self,Yp,dens,fields):
        s_field,v_field,p_field,d_field = fields
        m_eff = self.m-s_field
        kF_n = self.get_kF((1-Yp)*dens)
        kF_p = self.get_kF(Yp*dens)
        kF_e = np.piecewise(kF_p,[kF_p<=np.sqrt(self.m_m**2-self.m_e**2),kF_p>np.sqrt(self.m_m**2-self.m_e**2)],[kF_p,self.muon_equil(kF_p)])
        return np.sqrt(kF_n**2+(m_eff+.5*d_field)**2)-(np.sqrt(kF_p**2+(m_eff-.5*d_field)**2)+np.sqrt(kF_e**2+self.m_e**2)+p_field),kF_e

    def get_Yp(self,dens,Y0=0):
        """
        Solves beta-equilibrium and charge equilibrium conditions for the proton fraction.
        """
        min_func = lambda Y: self.beta_equil(Y,dens,self.solve_mesons(dens,Y))[0]
        Y = root(min_func,Y0).x
        return Y[0],self.beta_equil(Y[0],dens,self.solve_mesons(dens,Y[0]))[1]

    def get_rhoC(self):
        """
        Solves for the core-crust transition density using RPA.
        Returns the cc-density in fm^-3
        """

        q_a = np.linspace(1,400)
        d_i = .16
        h = .01
        while h>1e-5:
            d = d_i*self.hbc3
            Yp,kFe = self.get_Yp(d)
            fields = self.solve_mesons(d,Yp)
            Ye = kFe**3/3/np.pi**2/d
            eps = np.array(list(map(lambda q:self.delt_dielectric(q,d,Yp,Ye,*fields),q_a)))
            if np.any(eps <= 0):
                d_i += h
                h /= 10
            else:
                d_i -= h
                if d_i<0:
                    print("No crust-core determined. Breaking now.")
                    return 0

        rho_c = np.round(d_i,4) #don't keep the trailing zeros
        # print("Core-crust transition is " + str(rho_c) + " fm^-3")
        return rho_c

    #energy density integral
    def en_int(self,kF,m_eff):
        return 1/8/np.pi**2*(kF*np.sqrt(kF**2+m_eff**2)*(2*kF**2+m_eff**2)-m_eff**4*np.arcsinh(kF/m_eff))

    #pressure integral
    def pre_int(self,kF,m_eff):
        return 1/24/np.pi**2*(kF*(2*kF**2-3*m_eff**2)*np.sqrt(kF**2+m_eff**2)+3*m_eff**4*np.arcsinh(kF/m_eff))

    def e_QHD(self,dens,Yp,kFe,*fields):
        """
        Energy density of neutron star matter. Returns E/V in MeV^4
        """
        s_field,v_field,p_field,d_field = fields
        m_eff = self.m-s_field
        kF_n = self.get_kF((1-Yp)*dens)
        kF_p = self.get_kF(Yp*dens)
        kF_e = kFe
        kF_m = (kF_p**3-kF_e**3)**(1/3)

        #scalar-field terms
        s_terms = 1/6*self.kappa*(s_field)**3+1/24*self.lambd*(s_field)**4+1/2*self.ms**2/self.gs2*(s_field)**2
        #vector field energy terms
        v_terms = v_field*dens-1/2*self.mv**2/self.gv2*(v_field)**2-self.zeta/24*v_field**4
        #rho field energy terms
        r_terms = -1/2*self.mp**2/self.gp2*(p_field)**2+1/2*p_field*(2*Yp-1)*dens
        #delta field energy terms
        d_terms = np.piecewise(self.gd2, [self.gd2!=0.,self.gd2==0.], [1/2*d_field**2*self.md**2/self.gd2,0.0])
        #cross terms
        c_terms = -self.lambda_v*v_field**2*p_field**2-(self.lambda_1+self.lambda_2*s_field)*s_field*v_field**2
        c_terms += -(self.lambda_3+self.lambda_s*s_field)*s_field*p_field**2
        #particle energy terms
        neu_energy = self.en_int(kF_n,m_eff+.5*d_field)
        pro_energy = self.en_int(kF_p,m_eff-.5*d_field)
        ele_energy = self.en_int(kF_e,self.m_e)
        mu_energy = self.en_int(kF_m,self.m_m)

        return d_terms + s_terms + r_terms + neu_energy + pro_energy + v_terms + c_terms + ele_energy + mu_energy

    def p_QHD(self,dens,Yp,kFe,*fields):
        """
        Pressure of neutron star matter. Returns P in MeV^4
        """

        s_field,v_field,p_field,d_field = fields
        m_eff = self.m-s_field
        kF_n = self.get_kF((1-Yp)*dens)
        kF_p = self.get_kF(Yp*dens)
        kF_e = kFe
        kF_m = (kF_p**3-kF_e**3)**(1/3)


        #scalar self-interaction terms
        s_terms = -1/6*self.kappa*(s_field)**3-1/24*self.lambd*(s_field)**4-1/2*self.ms**2/self.gs2*(s_field)**2
        #vector field energy terms
        v_terms = 1/2*self.mv**2/self.gv2*(v_field)**2+self.zeta/24*v_field**4
        #rho field energy terms
        r_terms = 1/2*self.mp**2/self.gp2*(p_field)**2
        #delta field terms
        d_terms = np.piecewise(self.gd2, [self.gd2!=0.,self.gd2==0.], [-1/2*d_field**2*self.md**2/self.gd2,0.0])
        #cross terms
        c_terms = self.lambda_v*v_field**2*p_field**2+(self.lambda_1+self.lambda_2*s_field)*s_field*v_field**2
        c_terms += (self.lambda_3+self.lambda_s*s_field)*s_field*p_field**2
        #particle energy terms
        neu_energy = self.pre_int(kF_n,m_eff+.5*d_field)
        pro_energy = self.pre_int(kF_p,m_eff-.5*d_field)
        ele_energy = self.pre_int(kF_e,self.m_e)
        mu_energy = self.pre_int(kF_m,self.m_m)

        return d_terms + r_terms + s_terms + neu_energy + pro_energy + v_terms + c_terms + ele_energy + mu_energy

    def p_QHD_kaons(self,dens,Yp,kFe,kFm,kFn,kFp,*fields):
        """
        Pressure of neutron star matter. Returns P in MeV^4
        """

        s_field,v_field,p_field,d_field = fields
        m_eff = self.m-s_field
        kF_n = kFn
        kF_p = kFp
        kF_e = kFe
        kF_m = kFm


        #scalar self-interaction terms
        s_terms = -1/6*self.kappa*(s_field)**3-1/24*self.lambd*(s_field)**4-1/2*self.ms**2/self.gs2*(s_field)**2
        #vector field energy terms
        v_terms = 1/2*self.mv**2/self.gv2*(v_field)**2+self.zeta/24*v_field**4
        #rho field energy terms
        r_terms = 1/2*self.mp**2/self.gp2*(p_field)**2
        #delta field terms
        d_terms = np.piecewise(self.gd2, [self.gd2!=0.,self.gd2==0.], [-1/2*d_field**2*self.md**2/self.gd2,0.0])
        #cross terms
        c_terms = self.lambda_v*v_field**2*p_field**2+(self.lambda_1+self.lambda_2*s_field)*s_field*v_field**2
        c_terms += (self.lambda_3+self.lambda_s*s_field)*s_field*p_field**2
        #particle energy terms
        neu_energy = self.pre_int(kF_n,m_eff+.5*d_field)
        pro_energy = self.pre_int(kF_p,m_eff-.5*d_field)
        ele_energy = self.pre_int(kF_e,self.m_e)
        mu_energy = self.pre_int(kF_m,self.m_m)

        return d_terms + r_terms + s_terms + neu_energy + pro_energy + v_terms + c_terms + ele_energy + mu_energy




    def get_core_EOS(self,rho_C):
        E,P,mu_e,mu_n,Yp,fields = [],[],[],[],[],[]
        dens = np.logspace(np.log10(rho_C),np.log10(3),self.N_RMF)*self.hbc**3
        N = dens/self.hbc**3;fields_temp = [0,0,0,0]
        fields = np.zeros((len(dens),4))
        Yp_temp = 0
        for i in range(len(dens)):
            Yp_temp,kFe = self.get_Yp(dens[i],Yp_temp)
            fields[i,:] = self.solve_mesons(dens[i],Yp_temp,fields_temp)
            fields_temp = fields[i,:]
            mu_p_temp,mu_n_temp = self.get_chem_pot(dens[i],Yp_temp)
            Yp.append(Yp_temp)
            mu_n.append(mu_n_temp)
            #print(N[i],fields[i,:])
            E.append(self.e_QHD(dens[i],Yp_temp,kFe,*fields_temp))
            P.append(self.p_QHD(dens[i],Yp_temp,kFe,*fields_temp))
            mu_e.append(np.sqrt(kFe**2+self.m_e**2))
        E = np.array(E)/self.hbc3
        P = np.array(P)/self.hbc3
        mu_e = np.array(mu_e)
        mu_n = np.array(mu_n)
        Yp = np.array(Yp)
        return E,P,N,mu_e,mu_n,Yp,fields

    def get_chem_pot(self,dens,Yp):
        """
        Solves for the proton and neutron chemical potential.
        Input density in MeV^3 and proton fraction.
        Outputs: (mu_p, mu_n)
        """
        S,V,B,D = self.solve_mesons(dens,Yp)
        Mp = self.m-S-.5*D #proton effective mass
        Mn = self.m-S+.5*D #neutron effective mass
        kFp = self.get_kF(dens*Yp)
        kFn = self.get_kF(dens*(1-Yp))

        mu_p = np.sqrt(kFp**2+Mp**2)+V+.5*B
        mu_n = np.sqrt(kFn**2+Mn**2)+V-.5*B

        return mu_p,mu_n

    def get_chem_pot_kaons(self,Yp,kFn,kFp,fields):
        """
        Solves for the proton and neutron chemical potential.
        Input density in MeV^3 and proton fraction.
        Outputs: (mu_p, mu_n)
        """
        S,V,B,D = fields
        Mp = self.m-S-.5*D #proton effective mass
        Mn = self.m-S+.5*D #neutron effective mass

        mu_p = np.sqrt(kFp**2+Mp**2)+V+.5*B
        mu_n = np.sqrt(kFn**2+Mn**2)+V-.5*B

        return mu_p,mu_n



#ms,gs2,gv2,gp2,gd2,kappa,lambd,zeta,lambda_v = np.loadtxt('FSUGold.inp')        #read in the parameter file
#import matplotlib.pyplot as plt
#test = EOS_set(N_RMF=200)                                                        #declare the EOS class
#plt.plot(test.mu_n_vec/3,test.P_vec)
#plt.show()
#rho_C = test.get_rhoC()                                                         #solves for the core-crust transition density

#import matplotlib.pyplot as plt
#plt.plot(test.mu_n_vec/3,test.P_vec)
#plt.show()


#mup,mun = test.get_chem_pot(.15*test.hbc**3, .0614494)                               #input density and Yp to get
#print(mup,mun)
