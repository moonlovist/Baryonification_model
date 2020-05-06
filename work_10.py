from param import *
from adiabadic import *
from scipy.integrate import quad, quadrature, cumtrapz
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sympy import diff, symbols
from scipy.interpolate import splev, splrep
from scipy.misc import derivative
import time
from time import perf_counter
import numpy.fft as fft
import scipy.special as sp

def c_200(m_200):
     """
     Concentrations form Dutton+Maccio (2014)
     c200 (200 times RHOC)
     Assumes PLANCK cosmology
     """
     #A = 0.520 + (0.905-0.520)*np.exp(-0.617*z**1.21)
     A = 0.905
     B = -0.101 + 0.026*z
     return 10.0**A*(m_200/(1.0 * 10**12))**(B)

def r_200(M_200):
    return (M_200 * 3 / (4.0 * np.pi *200 * rho_critical)) ** (1/3.0)

def mass_func():
    return np.logspace(11, 18, 100)

def k_func(value):
    if value == True:
        k = np.logspace(-4, 3, 100)
        return k
    elif value == False:
        k = np.logspace(-4, 3, 100)
        return k

def rho_nfw(r, M_200):
    c = c_200(M_200)
    r_s = r_200(M_200)/c
    #delta_c = M_0 / (rho_critical * 16 * np.pi * r_s ** 3)
    global rho_nfw_0
    r_t = e * r_200(M_200)
    x = r/r_s
    y = r/r_t
    t = r_t / r_s
    m_nfw = (t**2 / (2*(1+t**2)**3)) * ((c*(c-2*t**6 + c*(1-3*c)*t**4 + c**2 + 2*(1+c-c**2)*t**2)/((1+c)*(t**2 + c**2))) + t*(6*t**2 - 2)*np.arctan(c/t) + t**2*(t**2 - 3)*np.log(t**2*(1+c)**2 / (t**2 + c**2)))
    rho_nfw_0 = M_200/(4 * np.pi * m_nfw * (r_200(M_200)/c)**3)
    return rho_nfw_0 /((x * (1 + x)**2) * (1 + y**2)**2)

def M_nfw(r_M, M):
    #func = lambda s: rho_nfw(s, M) * s ** 2 * 4 * np.pi
    M_1 = []
    for r in r_M:
        l = np.linspace(0.0001, r, 100)
        M_1.append(np.trapz(rho_nfw(l, M) * l ** 2 * 4 * np.pi, l))
    M_1 = np.array(M_1)
    return M_1

def M_total(M_200):
    r_vir = r_200(M_200)
    if type(M_200) is np.float64:
        r_MM = np.logspace(-5, 1, 200)
        total_masse = np.trapz((rho_nfw(r_MM, M_200) * r_MM ** 2 * 4 * np.pi), r_MM)
        return total_masse
    elif type(M_200) is np.ndarray:
        total_masse = []
        r_MM = np.logspace(-6,2,100)
        for M in M_200:
            #np.log(splev(M, r_func))
            total_masse.append(np.trapz((rho_nfw(r_MM, M) * r_MM ** 2 * 4 * np.pi), r_MM))
        return total_masse
    else:
        print("The type of M_200 is not float64 or ndarray, please check the type")

def M_total_2(r,M_200):
    c = c_200(M_200)
    r_s = r_200(M_200) / c
    r_t = e * r_200(M_200)
    x = r / r_s
    y = r / r_t
    t = r_t / r_s
    m_nfw = (t ** 2 / (2 * (1 + t ** 2) ** 3)) * ((c * (
                c - 2 * t ** 6 + c * (1 - 3 * c) * t ** 4 + c ** 2 + 2 * (1 + c - c ** 2) * t ** 2) / (
                                                               (1 + c) * (t ** 2 + c ** 2))) + t * (
                                                              6 * t ** 2 - 2) * np.arctan(c / t) + t ** 2 * (
                                                              t ** 2 - 3) * np.log(
        t ** 2 * (1 + c) ** 2 / (t ** 2 + c ** 2)))
    rho_nfw_0 = M_200 / (4 * np.pi * m_nfw * (r_200(M_200) / c) ** 3)
    m_nfw_2 = (2 * t**4 * (t**2 - 3)*np.log(t) - t**2 * (3*t**2 -1)*(t**2 - np.pi*t +1))/(2*(1+t**2)**3)
    return 4 * np.pi * m_nfw_2 * rho_nfw_0 * (r_200(M_200) / c) ** 3


def rho_cga(r_2, M_200, yita_cga):
    zeta = 1.376
    R_h = 0.015 * r_200(M_200)
    f_cga = 2 * NN * ((M_200 / M_1) ** (-zeta) + (M_200 / M_1) ** (yita_cga)) ** (-1.0)
    #f_cga = A * (M_1 / M_200)**(yita_cga)
    return f_cga * splev(M_200, get_M_total) * np.exp(-(r_2 / (2 * R_h))**2) / (4 * R_h * r_2 **2 * np.pi**1.5)#

def rho_star(r_2, M_200, yita_star):
    zeta = 1.376
    R_h = 0.015 * r_200(M_200)
    f_star = 2 * NN * ((M_200 / M_1) ** (-zeta) + (M_200 / M_1) ** (yita_star)) ** (-1.0)
    #f_cga = A * (M_1 / M_200)**(yita_cga)
    return f_star * splev(M_200, get_M_total) * np.exp(-(r_2 / (2 * R_h))**2) / (4 * R_h * r_2 **2 * np.pi**1.5)#

def rho_gas(r_2, M_200, M_c, mu, yita_star):
    #t1_start = perf_counter()
    theta_co = 0.1
    zeta = 1.376
    #theta_ej = 4 # a free parameter in the baryonic correction model
    r_co = theta_co * r_200(M_200)
    r_ej = theta_ej * r_200(M_200)
    u = r_2 / r_co
    v = r_2 / r_ej
    beta = 3 - (M_c / M_200)**mu
    global f_gas
    f_star = 2 * NN * ((M_200 / M_1) ** (-zeta) + (M_200 / M_1) ** (yita_star)) ** (-1.0)
    #f_star = A * (M_1 / M_200) ** yita_star
    f_gas = omega_b / omega_m - f_star
    #rho_gas_0 = f_gas * M_total * (4 * np.pi * quad(lambda r: r **2 / ((1 + r / r_co)**beta * (1 + (r / r_co)**2)**((7 - beta)/2)), 0, 10**15)[0])**(-1)
    func = lambda r: r ** 2 / ((1 + r / r_co) ** beta * (1 + (r / r_ej) ** 2) ** ((7 - beta) / 2))
    #rr = np.linspace(0,10**15,1000)
    rr = np.logspace(-3, 1, 100)
    rho_gas_0 = f_gas * splev(M_200, get_M_total) * ((4 * np.pi * np.trapz(func(rr),rr)) ** (-1))#
    rho_gas = rho_gas_0 / ((1 + u)**beta * (1 + v**2)**((7 - beta) / 2.0))
    #t1_stop = perf_counter()
    #print("Time used: gas", t1_stop-t1_start)
    return rho_gas

# define the mass of different model:
# question: how to define the M_200_special?

def rho_clm(r_2, M_200, yita_star):#, a, n
    M_200_list = mass_func()
    zeta = 1.376
    #f_star = A * (M_1 / M_200) ** yita_star
    #f_cga = A * (M_1 / M_200) ** (yita_cga)
    f_cga = 2 * NN * ((M_200 / M_1) ** (-zeta) + (M_200 / M_1) ** (yita_cga)) ** (-1.0)
    f_star = 2 * NN * ((M_200 / M_1) ** (-zeta) + (M_200 / M_1) ** (yita_star)) ** (-1.0)
    f_clm = omega_dm / omega_m + f_star - f_cga
    #print(get_M_nfw[np.where(M_200_list == M_200)])
    M_fnw_func_2 = splrep(r_2, get_M_nfw[np.where(M_200_list == M_200)][0])
    dif_clm_2 = splev(r_2, M_fnw_func_2, der=1)
    return (f_clm * dif_clm_2) / (4 * np.pi * r_2 **2)

def rho_dmb(r, M_200, M_c, mu, yita_star, yita_cga):
    return rho_gas(r, M_200, M_c, mu, yita_star) + rho_cga(r, M_200, yita_cga) + rho_clm(r, M_200, yita_star)

def integrat_dmo(r, M_200, k):
    u = rho_nfw(r,M_200) * 4 * np.pi * r**2 *np.sin(k * r)/ (k*r*splev(M_200, get_M_total))
    return u#(*4*np.pi*rho_nfw(r,M_200)*r_200(M_200)**3 /3.0)

def integrat_dmb(r, M_200, k, M_c, mu, yita_star, yita_cga):
    return rho_dmb(r, M_200, M_c, mu, yita_star, yita_cga) * 4 * np.pi * r**2 *np.sin(k * r) / (k * r * splev(M_200, get_M_total))

def sigma_2(M_200, P_lin):
    k = k_func(1)
    P_kk = interp1d(k, P_lin, kind='cubic')
    try:
        sigma_21 = []
        RR = (3 * M_200 / (4 * np.pi * rho_average)) ** (1 / 3.0)
        matrix = np.outer(RR,k)
        W = 3 * (np.sin(matrix) - matrix * np.cos(matrix)) / (matrix ** 3)
        f_mm = k ** 2 * P_kk(k) * W ** 2 / (2 * np.pi ** 2)
        for i in range(len(k)):
            sigma_21.append(np.trapz(f_mm[i], k))
        return np.array(sigma_21)**0.5
    except:
        R = (3 * M_200 / (4 * np.pi * rho_average)) ** (1 / 3.0)
        x = k * R
        W = 3 * (np.sin(x) - x * np.cos(x)) / (x**3)
        f_m = lambda k: k ** 2 * P_kk(k) * W ** 2 / (2 * np.pi ** 2) #W_KR abs
        #sigma_23 = quad(f_m, 0.1, 10)[0]  # integration of Fourier transformation, the integral range?
        sigma_23 = np.trapz(f_m(k), k)
        return sigma_23**0.5

def sigma_plus(M_200, P_lin):
    kk = np.logspace(-4,3,100)
    k_plus = [kk[0] - kk[0] / 10]
    k = np.append(np.array(k_plus), kk)
    P_kk = interp1d(k, P_lin, kind='cubic')
    sigma_21 = []
    RR = (3 * M_200 / (4 * np.pi * rho_average)) ** (1 / 3.0)
    matrix = np.outer(RR,k)
    W = 3 * (np.sin(matrix) - matrix * np.cos(matrix)) / (matrix ** 3)
    f_mm = k ** 2 * P_kk(k) * W ** 2 / (2 * np.pi ** 2)
    for i in range(len(k)):
        sigma_21.append(np.trapz(f_mm[i], k))
    return np.array(sigma_21)**0.5

def nu(M_200,P_lin):
    delta_sc = 1.686*(1+z)
    return delta_sc **2 / sigma_2(M_200,P_lin)**2

def deriv_nu(M_200, P_lin):
    k = k_func(1)
    P_kk = interp1d(k, P_lin, kind='cubic')
    delta_sc = 1.686*(1+z)
    d_nu = []
    sigma = splrep(M_200, sigma_2(M_200, P_lin))
    for m in np.array(M_200):
        dR = ((3 / (4 * np.pi * rho_average * m ** 2)) ** (1 / 3.0)) / 3.0
        R = (3 * m / (4 * np.pi * rho_average)) ** (1 / 3.0)
        x = k * R
        W_2 = (3 * (np.sin(x) - x * np.cos(x)) / (x ** 3))**2
        W_func = splrep(M_200, W_2)
        d_sigma = np.trapz(k**2 * P_kk(k) * splev(M_200,W_func,der=1)/(2 * np.pi**2),k)
        d_nu.append(- delta_sc**2 * d_sigma/(splev(m, sigma)**4))
    d_nu = np.array(d_nu)
    #d_nu[d_nu < 0] = 0.1
    return d_nu

def f_sigma(M_200, P_lin):
    '''
    A = 0.282
    a = 2.163
    b = 1.406
    c = 1.21
    sigma = sigma_2(M_200,P_lin)
    return A * ((sigma / b)**(-a) + 1) * np.exp(-c / sigma**2)
    '''
    P = 0.3
    a = 0.707
    A = 0.3222
    delta = 1.686
    sigma = sigma_2(M_200, P_lin)
    return A * (2 * a / np.pi)**0.5 * (1 + (sigma ** 2 / (a * delta**2)) ** P) * np.exp(-a * delta**2 / (2 * sigma**2)) * delta / sigma

def f_sigma_2(M_200, P_lin):
    P = 0.3
    a = 0.707
    A = 0.3222
    delta = 1.686
    sigma = sigma_2(M_200, P_lin)
    return  (2 / np.pi)**0.5 * np.exp(-a * delta**2 / (2 * sigma**2)) * delta / sigma

def fv(M_200,P_lin):
    P = 0.3#0.2536
    q = 0.707#0.7689
    A_p = 0.3222#0.3295
    v = nu(M_200,P_lin)
    return A_p * (1 + (q * v)**(-P)) * (q / (2 * np.pi * v )) * np.exp(-q * v / 2)

def bias(M_200, P_lin):
    q = 0.707
    P = 0.3
    a_2 = -17/21.0
    delta_sc = 1.686*(1+z)
    q_v = q * nu(M_200, P_lin)
    e_1 = (q_v - 1) / delta_sc
    e_2 = q_v * (q_v - 3) / delta_sc **2
    E_1 = 2 * P / (delta_sc * (1 + q_v ** P))
    E_2 = E_1 * (2 * e_1 + (1 + 2 * P)/delta_sc)
    b_1 = 1 + e_1 + E_1
    b_2 = 2 * (1 + a_2) * (e_1 + E_1) + e_2 + E_2
    return b_1

def P_hh(M_1, M_2, P_lin):
    P = []
    try:
        b_1_1, b_2_1 = bias(M_1, P_lin)
        b_1_2, b_2_2 = bias(M_2, P_lin)
        P_2h = b_1_1 * b_1_2 * P_lin
        return P_2h
    except:
        b_1_1, b_2_1 = bias(M_1, P_lin)
        for M in M_2:
            b_1_2, b_2_2 = bias(M, P_lin)
            P.append(b_1_1 * b_1_2 * P_lin)
        return P

def bias_func(M, P_lin):
    b_1 = bias(M, P_lin)
    b_2 = b_1
    matrix = np.outer(b_1, b_2)
    return matrix

def number_density_1h(M_200,P_lin):
    sigma_func = splrep(M_200, sigma_2(M_200, P_lin))
    dn_dm = - rho_average * f_sigma_2(M_200, P_lin) * splev(M_200, sigma_func, der=1) / (M_200 * splev(M_200, sigma_func))
    dn_dm_M = dn_dm
    n_func = dn_dm_M
    p_average_1 = np.trapz(n_func * M_200, M_200) / rho_average
    print("1h:",p_average_1)
    n_func = n_func / p_average_1
    print(np.trapz(n_func * M_200, M_200) / rho_average)
    return n_func

def number_density_2h(M_200,P_lin):
    sigma_func = splrep(M_200, sigma_2(M_200, P_lin))
    dn_dm = - rho_average * f_sigma_2(M_200, P_lin) * splev(M_200, sigma_func, der=1) / (M_200 * splev(M_200, sigma_func))
    dn_dm_M = dn_dm
    n_func = dn_dm_M
    p_average_1 = np.trapz(n_func * M_200 * bias(M_200, P_lin), M_200)/ rho_average
    print("2h:", p_average_1)
    n_func = n_func / p_average_1
    print(np.trapz(n_func * M_200 * bias(M_200, P_lin), M_200)/ rho_average)
    return n_func

# one halo term
def P_dmo(M_200,P_lin):
    t1_start = perf_counter()
    P = []
    kk = k_func(0)
    N = number_density_1h(M_200, P_lin)
    r_vir = r_200(M_200)
    r_range = np.logspace(-3, 1, 100)
    for k in kk:
        u_func = lambda r, M: integrat_dmo(r, M, k)
        r_s_func = splrep(M_200, r_vir)
        uu_1 = [np.trapz(u_func(r_range, M), r_range) for M in M_200]#np.log(splev(M, r_s_func))
        uu_1 = np.array(uu_1)
        uu_1[uu_1<0.001] = 0.001#/max(uu_test)
        uu = splrep(M_200, uu_1)
        P.append(np.trapz(N * splev(M_200, uu)**2 * M_200**2 / rho_average**2, M_200))
    t1_stop = perf_counter()
    print("dmo 1h Time used:", t1_stop - t1_start)
    return np.array(P)

def P_dmb(M_200,P_lin):
    t1_start = perf_counter()
    P = []
    kk = k_func(0)
    N = number_density_1h(M_200, P_lin)
    r_vir = r_200(M_200)

    for k in kk:
        u_func = lambda r,M: integrat_dmb(r, M, k, M_c, mu, yita_star, yita_cga)
        r_s_func = splrep(M_200, r_vir)
        uu_1 = np.array([np.trapz(u_func(np.logspace(-3, 1, 100), M), np.logspace(-3, 1, 100))for M in M_200])
        uu_1[uu_1 < 0.001] = 0
        uu = splrep(M_200, uu_1)#np.log(splev(M, r_s_func))
        P.append(np.trapz(N * splev(M_200, uu)**2 * M_200**2 / rho_average ** 2, M_200))
    t1_stop = perf_counter()
    print("dmb 1h Time used:", t1_stop - t1_start)
    return np.array(P)

# two halo term

def P_dmo_2h(M_200, P_lin, P_lin_1):
    P = []
    kk = k_func(0)
    r_vir = r_200(M_200)
    t1_start = perf_counter()
    N = number_density_2h(M_200, P_lin)
    for k in kk:
        P_1 = []
        u_func = lambda r, M: integrat_dmo(r, M, k)
        r_s_func = splrep(M_200, r_vir)
        uu_1 = [np.trapz(u_func(np.logspace(-3, 1, 100), M), np.logspace(-3, 1, 100)) for M in M_200]  # np.log(splev(M, r_s_func))
        uu_1 = np.array(uu_1)
        uu_1[uu_1 < 0.001] = 0
        #uu_1 = [np.trapz(u_func(np.linspace(0.00000001, splev(M, r_s_func), 100), M), np.linspace(0.00000001, splev(M, r_s_func), 100)) for M in M_200]
        uu = splrep(M_200, uu_1)
        p_hh = np.array(bias_func( M_200, P_lin_1))
        p_1 = p_hh * splev(M_200, uu) * N * M_200 / rho_average
        for i in range(len(M_200)):
            P_1.append(np.trapz(p_1[i], M_200))#
        P.append(np.trapz(P_1 * N * splev(M_200, uu) * M_200 / rho_average, M_200))
    t1_stop = perf_counter()
    print("dmo 2h Time used:", t1_stop - t1_start)
    return np.array(P) * P_lin

def P_dmb_2h(M_200,P_lin):
    P = []
    kk = k_func(0)
    r_vir = r_200(M_200)
    yita_adiabadic = np.loadtxt("adiabadic_yita.txt", unpack=True)
    N = number_density_2h(M_200, P_lin)
    t1_start = perf_counter()
    for k in kk:
        P_1 = []
        u_func = lambda r, M: integrat_dmb(r, M, k, M_c, mu, yita_star, yita_cga)
        r_s_func = splrep(M_200, r_vir)
        uu_1 = np.array([np.trapz(u_func(np.logspace(-3, 1, 100), M), np.logspace(-3, 1, 100)) for M in M_200])#np.log(splev(M, r_s_func))
        uu_1[uu_1 < 0.001] = 0
        uu = splrep(M_200, uu_1)
        p_hh = np.array(bias_func( M_200, P_lin_1))
        p_1 = p_hh * splev(M_200, uu) * N * M_200/ rho_average#
        #print(p_1)
        for i in range(len(M_200)):
            P_1.append(np.trapz(p_1[i], M_200))
        #print(P_1)
        P.append(np.trapz(P_1 * N * splev(M_200, uu) * M_200/ rho_average, M_200))
    t1_stop = perf_counter()
    print("dmb 2h Time used:", t1_stop - t1_start)
    return np.array(P) * P_lin

def P_lin_spec(value):
    P_lin = np.loadtxt("linear_powerspectrom/Pnw_2.dat", unpack=True)
    #P_lin = np.loadtxt("linear_powerspectrom/P_linear_EH_2.txt", unpack=True)
    P_k = interp1d(P_lin[0], P_lin[1], kind='cubic')
    k = k_func(value)
    P = P_k(k)
    return P

def M_func(r_M, M):
    M_1 = []
    for r in r_M:
        l = np.linspace(0.0001, r, 100)
        M_1.append(np.trapz(rho_nfw(l, M) * l ** 2 * 4 * np.pi, l))
    M_1 = np.array(M_1)
    return M_1

def define_Mass_function(M_200):
    global get_M_total
    global get_M_nfw
    get_M_total = splrep(M_200, M_total(M_200))
    r_2 = np.logspace(-3, 1, 100)
    yita_adiabadic = np.loadtxt("adiabadic_yita.txt", unpack=True)
    Mass_nfw = []
    for M in M_200:
        yita_1 = yita_adiabadic[np.where(M_200 == M)]
        r_3 = r_2/yita_1[0]
        Mass_nfw.append(M_nfw(r_3, M))
    get_M_nfw = np.array(Mass_nfw)

t2_start = perf_counter()

kk = k_func(1)
k = k_func(0)
P_lin = P_lin_spec(1)
P_lin_1 = P_lin_spec(0)

r = np.logspace(-3,1,len(kk))     # r used for dark matter only model [Mpc/h]
M_200 = mass_func()# M_solar / h
b_1 = bias(M_200, P_lin)

define_Mass_function(M_200)

def P_dmo_func():
    P_dmo_1hh = P_dmo(M_200, P_lin)
    P_dmo_2hh = P_dmo_2h(M_200, P_lin, P_lin_1)
    P_dmo_total = P_dmo_1hh + P_dmo_2hh
    np.savetxt('data_profiles/powspec_dmo_test.txt', np.transpose([k, P_dmo_1hh, P_dmo_2hh, P_dmo_total]), fmt='%.8g')
def P_dmb_func():
    P_dmb_1hh = P_dmb(M_200, P_lin)
    P_dmb_2hh = P_dmb_2h(M_200, P_lin)
    P_dmb_total = P_dmb_1hh + P_dmb_2hh
    np.savetxt('data_profiles/powspec_dmb_test.txt', np.transpose([k, P_dmb_1hh, P_dmb_2hh, P_dmb_total]), fmt='%.8g')


#M_c = 10**12#6.6 * 10 ** 13

rrr = np.logspace(-3, 1, len(M_200))
yita_adiabadic = np.loadtxt("adiabadic_yita.txt", unpack=True)

P_dmo_func()
P_dmb_func()

M_density = M_200[50]
print(M_density)
density_dmo = rho_nfw(r, M_density)# + rho_2h(r,M_density,P_lin_spec(1))
density_dmb = rho_dmb(r, M_density, M_c, mu, yita_star, yita_cga)
density_dmb_gas = rho_gas(r, M_density, M_c, mu, yita_star)
density_dmb_cga = rho_cga(r, M_density, yita_cga)
density_dmb_clm = rho_clm(r, M_density, yita_star)

#print(M_200)

name_1 = 11
for name in [M_200[0], M_200[20], M_200[40], M_200[60], M_200[80], M_200[99]]:
    density_dmbb = rho_dmb(r, name, M_c, mu, yita_star, yita_cga)
    density_dmbb_cga = rho_cga(r, name, yita_cga)
    density_dmbb_star = rho_cga(r, name, yita_star)
    density_dmbb_gas = rho_gas(r, name, M_c, mu, yita_star)
    density_dmbb_clm = rho_clm(r, name, yita_star)
    np.savetxt('data_profiles/density_dmb_%s.txt'%name_1, np.transpose([r, density_dmbb, density_dmbb_cga, density_dmbb_star, density_dmbb_gas, density_dmbb_clm]), fmt='%.8g')
    name_1 = name_1 + 1

np.savetxt('density_dmo.txt', np.transpose([r, density_dmo]), fmt='%.8g')
np.savetxt('density_dmb.txt', np.transpose([r, density_dmb_gas, density_dmb_cga, density_dmb_clm, density_dmb]), fmt='%.8g')

t2_stop = perf_counter()
print("Total Time used:", t2_stop - t2_start)
