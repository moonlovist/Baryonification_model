import numpy as np

e = 4
omega_dm = 0.272#0.233
omega_b = 0.048#0.0463
omega_m = 0.32#0.2793
H = 67#72 # Hubble constant (km /s / Mpc)
#G = 6.67430 * 10**(-11)# Gravitational constant (N⋅m2/kg2)
G = 4.299e-9#[Mpc/M_solar*(km/s)**2]

global A
#global M_1
global NN
global M_1
global M_c
global mu
global yita_star
global yita_cga
global theta_ej
global z
global rho_critical
global rho_average
A = 0.09
#M_1 = 2.5 * 10 ** 11  # unites: M_solar / h
NN = 0.0351
M_1 = 10.0**11.4351/0.704
M_c = 10**14
mu = 0.4
yita_cga = 0.6
theta_ej = 8
yita_star = 0.3
z = 0
rho_critical = 2.7755e11 #Msun/h/(Mpc/h)^3 #8.62 * 10**(-27) #kg/m3
rho_average = rho_critical * omega_m
print("use the default parameters")
