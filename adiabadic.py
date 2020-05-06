from param import *
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
import scipy.optimize
import pdb


rho_critical = 2.7755e11 #Msun/h/(Mpc/h)^3 #8.62 * 10**(-27) #kg/m3
rho_average = rho_critical * omega_m

#From work_9############################################################################################################
def c_200(m_200):
    A = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z ** 1.21)
    B = -0.101 + 0.026 * z
    return 10.0 ** A * (m_200 / (1.0 * 10 ** 12)) ** (B)
def r_200(M_200):
    return (M_200 * 3 / (4.0 * np.pi *200 * rho_critical)) ** (1/3.0)
def rho_nfw(r, M_200):
    try:
        c = c_200(M_200)
        r_s = r_200(M_200)/c
        r_t = e * r_200(M_200)
        #delta_c = M_0 / (rho_critical * 16 * np.pi * r_s ** 3)
        global rho_nfw_0
        #rho_nfw_0 = 4 * delta_c * rho_critical
        x = r / r_s
        y = r / r_t
        t = r_t / r_s
        m_nfw = (t ** 2 / (2 * (1 + t ** 2) ** 3)) * ((c * (c - 2 * t ** 6 + c * (1 - 3 * c) * t ** 4 + c ** 2 + 2 * (1 + c - c ** 2) * t ** 2) / ((1 + c) * (t ** 2 + c ** 2))) + t * (6 * t ** 2 - 2) * np.arctan(c / t) + t ** 2 * (
                                                                  t ** 2 - 3) * np.log(
            t ** 2 * (1 + c) ** 2 / (t ** 2 + c ** 2)))
        rho_nfw_0 = M_200 / (4 * np.pi * m_nfw * (r_200(M_200) / c) ** 3)

        return rho_nfw_0 /((x * (1 + x)**2) * (1 + y**2)**2)
    except:
        print("problem")
        return 0

def M_total(M_200):
    r_vir = r_200(M_200)
    if type(M_200) is np.float64:
        r_MM = np.logspace(-5, 1, 200)
        total_masse = np.trapz((rho_nfw(r_MM, M_200) * r_MM ** 2 * 4 * np.pi), r_MM)
        return total_masse
    elif type(M_200) is np.ndarray:
        total_masse = []
        r_MM = np.logspace(-6, 2, 100)
        for M in M_200:
            # np.log(splev(M, r_func))
            total_masse.append(np.trapz((rho_nfw(r_MM, M) * r_MM ** 2 * 4 * np.pi), r_MM))
        return total_masse
    else:
        print("The type of M_200 is not float64 or ndarray, please check the type")

def rho_cga(r_2, M_200, yita_cga):
    R_h = 0.015 * r_200(M_200)
    #f_cga = A * (M_1 / M_200)**(yita_cga)
    zeta = 1.376
    f_cga = 2 * NN * ((M_200 / M_1) ** (-zeta) + (M_200 / M_1) ** (yita_cga)) ** (-1.0)
    return f_cga * splev(M_200, get_M_total) * np.exp(-(r_2 / (2 * R_h))**2) / (4 * R_h * r_2 **2 * np.pi**1.5)

def rho_gas(r_2, M_200, M_c, mu, yita_star):
    zeta = 1.376
    theta_co = 0.1
    r_co = theta_co * r_200(M_200)
    r_ej = theta_ej * r_200(M_200)
    u = r_2 / r_co
    v = r_2 / r_ej
    beta = 3 - (M_c / M_200)**mu
    f_star = 2 * NN * ((M_200 / M_1) ** (-zeta) + (M_200 / M_1) ** (yita_star)) ** (-1.0)
    #f_star = A * (M_1 / M_200)**yita_star
    f_gas = omega_b / omega_m - f_star
    if f_gas<0:
        f_gas=0.0001
    func = lambda r_2: r_2 ** 2 / ((1 + r_2 / r_co) ** beta * (1 + (r_2 / r_ej) ** 2) ** ((7 - beta) / 2))
    rr = np.logspace(-3, 1, 100)
    rho_gas_0 = f_gas * splev(M_200, get_M_total) * ((4 * np.pi * np.trapz(func(rr),rr)) ** (-1))
    rho_gas = rho_gas_0 / ((1 + u)**beta * (1 + v**2)**((7 - beta) / 2.0))
    return rho_gas

def define_Mass_function(M_200):
    global get_M_total
    get_M_total = splrep(M_200, M_total(M_200))

#Generate all density profiles##########################################################################################
M_200 = np.logspace(11, 18, 100)
r = np.logspace(-3,1,len(M_200))
define_Mass_function(M_200)

#t2_start = perf_counter()
Matrix = np.outer(M_200, r)
density_inte_dmo = Matrix*0
density_inte_cga = Matrix*0
density_inte_gas = Matrix*0
l = []
M_inte_nfw = Matrix*0
M_inte_gas = Matrix*0
M_inte_cga = Matrix*0
le = len(M_200)
for i in range(le):
    l.append(np.logspace(-8, np.log(r[i]), 100))

for i in range(le):
    for j in range(le):
        density_inte_dmo[i][j] = rho_nfw(r[i],M_200[j])
        density_inte_gas[i][j] = rho_gas(r[i],M_200[j], M_c, mu, yita_star)
        density_inte_cga[i][j] = rho_cga(r[i],M_200[j], yita_cga)

        M_inte_nfw[i][j] = np.trapz(density_inte_dmo[i][j] * l[i] ** 2 * 4 * np.pi, l[i])
        M_inte_gas[i][j] = np.trapz(density_inte_gas[i][j] * l[i] ** 2 * 4 * np.pi, l[i])
        M_inte_cga[i][j] = np.trapz(density_inte_cga[i][j] * l[i] ** 2 * 4 * np.pi, l[i])

M_inte_nfw = np.transpose(M_inte_nfw)
M_inte_gas = np.transpose(M_inte_gas)
M_inte_cga = np.transpose(M_inte_cga)
Z = []
a = 0.68
n = 1
zeta = 1.376
f_inte_cga = 2 * NN * ((M_200 / M_1) ** (-zeta) + (M_200 / M_1) ** (yita_cga)) ** (-1.0)
f_inte_star = 2 * NN * ((M_200 / M_1) ** (-zeta) + (M_200 / M_1) ** (yita_star)) ** (-1.0)
f_inte_gas = omega_b / omega_m - f_inte_star
f_inte_clm = omega_dm / omega_m + f_inte_star - f_inte_cga

for j in range(le):
    M_inte_cga_f = splrep(r, M_inte_cga[j])
    M_inte_gas_f = splrep(r, M_inte_gas[j])
    M_inte_nfw_f = splrep(r, M_inte_nfw[j])
    func_1 = lambda k: a * ((splev(r, M_inte_nfw_f) / (
                f_inte_clm[j] * splev(r, M_inte_nfw_f) + f_inte_cga[j] * splev(r * k, M_inte_cga_f) + f_inte_gas[j] * splev(
            r * k, M_inte_gas_f))) ** n - 1) + 1 - k
    Z.append(scipy.optimize.broyden1(func_1, r*0+2, f_tol=1e-10))
np.savetxt('adiabadic_yita_2.txt', np.transpose(Z), fmt = '%.8g')
print("adiabadic run successfully")
