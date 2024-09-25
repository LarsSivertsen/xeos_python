import numpy as np
from scipy import interpolate
from scipy.integrate import cumtrapz,solve_ivp

CONV_MeV_fm3_to_g_cm3 = 1.78266181*1e12#1.78266181e-36 * 1e48
CONV_MeV_fm3_to_dyn_cm2 = 1.78266181*2.99792458**2*1e32#1.78266181e-36*2.99792458e8**2*1e52


G = 6.6743 * 10**(-8)      # Newton's gravitational constant in cgs units
c = 2.99792458 * 10**10    # speed of light in cgs units
M_sun = 1.476 * 10**(5)    # Mass of the sun in geometrized units


def tovrhs(t,y,eos1,eos2,eos3):

    r, m, H, b = y

    eps = eos1(t)
    p = eos2(t)
    c2_s = eos3(t)

    A = 1.0 / (1.0 - 2.0 * m / r)
    C1 = 2.0 / r + A * (2.0 * m / (r * r) + 4.0 * np.pi * r * (p - eps))
    C0 = A * ( -(2) * (2 + 1) / (r * r)  + 4.0 * np.pi * (eps + p)/c2_s
              + 4.0 * np.pi * (5.0 * eps + 9.0 * p))- np.power(2.0 * (m + 4.0 * np.pi * r * r * r * p) / (r * (r - 2.0 * m)), 2.0)


    dr_dh = -r*(r-2*m)/(m + 4* np.pi * r**3 * p )

    dm_dh = 4 * np.pi * r**2 * eps * dr_dh

    dH_dh = b * dr_dh

    db_dh = -(C0 * H + C1 * b) * dr_dh

    return [dr_dh,dm_dh, dH_dh, db_dh]





def tov_solve(epsilon,press,c2_s,rtol=1.e-6,atol=1.e-5,TOV_limit=True, h_c_grid = 'coarse'):

    epsilon = epsilon  *   CONV_MeV_fm3_to_g_cm3       * G * M_sun**2 / c**2
    press   = press    *   CONV_MeV_fm3_to_dyn_cm2     * G * M_sun**2 / c**4

    enthalpy = cumtrapz(1/(epsilon+press), press, initial=0)

    enthalpy = np.sort(enthalpy)

    max_enthalpy = np.amax(enthalpy)


    spl1 = interpolate.splrep(enthalpy, np.log(epsilon),k=1)
    spl2 = interpolate.splrep(enthalpy, np.log(press),k=1)
    spl3 = interpolate.splrep(enthalpy, c2_s, k=1)

    def eos1(h):
        return np.exp(interpolate.splev(h, spl1, der=0))

    def eos2(h):
        return np.exp(interpolate.splev(h, spl2, der=0))

    def eos3(h):
        return interpolate.splev(h, spl3, der=0)

    if h_c_grid == 'coarse':
        central_enthalpies = np.geomspace(0.01,max_enthalpy,125)
    else:
        central_enthalpies = np.linspace(0.01,max_enthalpy,500)


    Radius = np.zeros_like(central_enthalpies)
    Mass = np.zeros_like(central_enthalpies)
    Lambda = np.zeros_like(central_enthalpies)


    for i,h_c in enumerate(central_enthalpies):

        r0 = 1.e-3

        m0 = 4/3 * np.pi * r0**3 * eos1(h_c)

        H0 = r0**2

        b0 = 2.0 * r0

        initial = r0, m0, H0, b0

        sol1 = solve_ivp(tovrhs, (h_c, 0.0), initial, args=(eos1,eos2,eos3),method='LSODA',
                        rtol=rtol,atol=atol)


        R = sol1.y[0, -1]
        M = sol1.y[1, -1]
        C = M/R

        H_surf = sol1.y[2, -1]
        b_surf = sol1.y[3, -1]

        Y = R * b_surf / H_surf

        fac1 = (8.0 / 5.0) * np.power(1 - 2 * C, 2.0)* np.power(C, 5.0)* (2 * C * (Y - 1) - Y + 2)
        fac2 = 2* C* ( 4 * (Y + 1) * np.power(C, 4) + (6 * Y - 4) * np.power(C, 3) + (26 - 22*Y) *C*C + 3 *(5*Y -8)* C- 3 * Y+ 6)
        fac3 = -3 * np.power(1 - 2 * C, 2) * (2 * C * (Y - 1) - Y + 2) * np.log(1.0 / (1 - 2 * C))
        k2   = fac1 / (fac2+fac3)

        Lambda[i] = 2. / 3. * k2 * np.power(C, -5.)
        Radius[i] = R * M_sun * 10**(-5)
        Mass[i]   = M

        if TOV_limit:
            if Mass[i] < Mass[i-1]:
                Mass = Mass[:i]
                Radius = Radius[:i]
                Lambda = Lambda[:i]
                break

    p_c = eos2(central_enthalpies)/(CONV_MeV_fm3_to_dyn_cm2*G*M_sun**2/c**4)  # Central pressures in MeV fm-3
    p_c = p_c[:len(Mass)]

    return Radius,Mass,Lambda,p_c
