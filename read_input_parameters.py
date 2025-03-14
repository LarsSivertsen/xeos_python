def get_number_before_number_sign(string):
    new_string = ""
    i=0
    while(string[i]!="#" and string[i]!="\n"):
        if(string[i]!="''" and string[i]!='"'):
            new_string+=string[i]
        i+=1
    return new_string

def get_range(string):
    new_string_lower = ""
    i=0
    while (string[i]!=","):
        new_string_lower+=string[i]
        i+=1
    i+=1
    new_string_upper = get_number_before_number_sign(string[i:])
    return new_string_lower,new_string_upper

def read_parameters_main():
    with open("input_parameters_main.txt") as file:
        N = int(get_number_before_number_sign(file.readline()))
        run_number = int(get_number_before_number_sign(file.readline()))
        TOV = bool(int(get_number_before_number_sign(file.readline())))
        TOV_limit = bool(int(get_number_before_number_sign(file.readline())))
        N_low_dens = int(get_number_before_number_sign(file.readline()))
        N_CFL = int(get_number_before_number_sign(file.readline()))
        rho_max_low_dens = float(get_number_before_number_sign(file.readline()))
        rho_max_high_dens = float(get_number_before_number_sign(file.readline()))
        dmu_q = float(get_number_before_number_sign(file.readline()))
        dmu_q_factor = float(get_number_before_number_sign(file.readline()))
        B_min_string,B_max_string = get_range(file.readline())
        B_range = [float(B_min_string),float(B_max_string)]
        Delta_min_string,Delta_max_string = get_range(file.readline())
        Delta_range = [float(Delta_min_string),float(Delta_max_string)]
        m_s_min_string,m_s_max_string = get_range(file.readline())
        m_s_range = [float(m_s_min_string),float(m_s_max_string)]
        c_min_string,c_max_string = get_range(file.readline())
        c_range = [float(c_min_string),float(c_max_string)]
        distribution_B = str(get_number_before_number_sign(file.readline()).strip(" "))
        distribution_Delta = str(get_number_before_number_sign(file.readline()).strip(" "))
        distribution_m_s = str(get_number_before_number_sign(file.readline()).strip(" "))
        distribution_c = str(get_number_before_number_sign(file.readline()).strip(" "))
        RMF_filename = str(get_number_before_number_sign(file.readline()).strip(" "))
        eos_name = str(get_number_before_number_sign(file.readline()).strip(" "))



        print("N:",N)
        print("run_number:",run_number)
        print("TOV:",TOV)
        print("TOV_limit:",TOV_limit)
        print("N_low_dens",N_low_dens)
        print("N_CFL:",N_CFL)
        print("rho_max_low_dens:",rho_max_low_dens)
        print("rho_max_high_dens:",rho_max_high_dens)
        print("dmu_q:",dmu_q)
        print("dmu_q_factor:",dmu_q_factor)
        print("B_range:",B_range)
        print("Delta_range:",Delta_range)
        print("m_s_range:",m_s_range)
        print("c_range:",c_range)
        print("distribution_B:",distribution_B)
        print("distribution_Delta:",distribution_Delta)
        print("distribution_m_s:",distribution_m_s)
        print("distribution_c:",distribution_c)
        print("RMF_filename:",RMF_filename)
        print("eos_name:",eos_name)

        return (N,
                run_number,
                TOV,
                TOV_limit,
                N_low_dens,
                N_CFL,
                rho_max_low_dens,
                rho_max_high_dens,
                dmu_q,
                dmu_q_factor,
                B_range,
                Delta_range,
                m_s_range,
                c_range,
                distribution_B,
                distribution_Delta,
                distribution_m_s,
                distribution_c,
                RMF_filename,
                eos_name)
