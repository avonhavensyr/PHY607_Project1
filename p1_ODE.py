import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import methods as meth

# Adding the rcParams from my undergrad research plotting file
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 11

"""
PROJECT I
Draft: Includes everything for the ODE problem in one file. 
Separation between module and main file will be implemented later.
"""

"""
ODE Solver: Coupled Harmonic Oscillator
"""

# CONSTANTS
m1 = 1
m2 = 10
k = 1

# INITIAL CONDITIONS

# Initial Positions
x10 = 1
x20 = 1
# Initial Velocities
v10 = 1
v20 = 1
# Initial Time + Time Step
t0 = 0

def f(x, xi, m):
    """
    Equation of motion for one mass in the coupled oscillator system

    Keyword Arguments:
    x (int/float): position of reference mass
    xi (int/float): position of coupled mass
    m (int/float): mass of reference mass
    """
    dvi = ((1 / m) * ((-2 * k * x) + (k * xi)))
    return dvi

def coupledOscillator(m1, m2, x10, x20, v10, v20, k, t, test = False):
    """
    Analytical solution to the coupled oscillator as a superposition of 
    normal mode solutions

    Keyword Arguments:
    m1 (int/float): mass of mass 1
    m2 (int/float): mass of mass 2
    x10 (int/float): initial position of mass 1
    x20 (int/float): initial position of mass 2
    v10 (int/float): initial velocity of mass 1
    v20 (int/float): initial velocity of mass 2
    k (int/float): spring constant
    t (int/float): current time
    test (boolean): returns extra values if True-- default is False
    """
    # Setting constants to make typing final variables less tedious
    m_sum = m1 + m2
    m_pro = m1 * m2
    root = np.sqrt((m1**2) + (m2**2) - m_pro)
    omega_p = (k/m_pro) * (m_sum + root)
    omega_n = (k/m_pro) * (m_sum - root)   

    # Ratio of the two amplitudes for the omega_p and omega_n frequencies
    r_p = 2 - ((m1 * omega_p) / k)
    r_n = 2 - ((m1 * omega_n) / k)

    # Cosine co-efficients
    C1 = (m2 / (2 * root)) * ((r_n * x10) - x20)
    C2 = (m2 / (2 * root)) * (x20 - (r_p * x10))
    # Sine co-efficients
    S1 = (m2 / (2 * root)) * ((r_n * v10) - v20) / (np.sqrt(omega_p))
    S2 = (m2 / (2 * root)) * (v20 - (r_p * v10)) / (np.sqrt(omega_n))

    # Once again pre-calculating terms because writing everything inline got too long
    cos_p = np.cos(np.sqrt(omega_p) * t)
    cos_n = np.cos(np.sqrt(omega_n) * t)
    sin_p = np.sin(np.sqrt(omega_p) * t)
    sin_n = np.sin(np.sqrt(omega_n) * t)

    # Final general solutions
    x1 = (C1 * cos_p) + (S1 * sin_p) + (C2 * cos_n) + (S2 * sin_n)
    x2 = (r_p * ((C1 * cos_p) + (S1 * sin_p))) + (r_n * ((C2 * cos_n) + (S2 * sin_n)))
    if test:
        return (x1, x2, omega_p, omega_n)
    else:
        return (x1, x2)

def coupledEnergy(mi, mj, vi, vj, xi, xj):
    """
    Function to compute the potential, kinetic, and total energy of the coupled oscillator

    Keyword arguments:
    mi (int/float): mass of mass i
    mj (int/float): mass of mass j
    vi (int/float): initial/current velocity value of mass i
    vj (int/float): initial/current velocity value of mass j
    xi (int/float): initial/current position value of mass i
    xj (int/float): initial/current position value of mass j
    """
    # Potential energy (taken as -grad(-kxi)) for masses mi and mj
    Uij = k * ((xi**2) + (xj**2) - (xi * xj))
    # Kinetic energy for masses mi and mj
    Tij = (0.5 * mi * (vi**2)) + (0.5 * mj * (vj**2))
    # Total Energy
    Eij = Uij + Tij
    return np.array([Uij, Tij, Eij])


# Put F into a form that will work for solve_ivp method
def fNew(t, s, mi, mj):
    """
    Function that converts the state of the system into a form that 
    is usable for SciPy's solve_ivp RK45 method

    Keyword Arguements:
    t (float/int): current time at position n
    s (array): current state vector of position n-- takes the form [xi, xj, vi, vj]
    mi (int/float): mass of mi
    mj (inf/float): mass of mj
    """
    # Separate state vector into variables
    xi, xj, vi, vj = s
    # Use function f to get the accelerations of mi and mj
    dvi = f(xi, xj, mi)
    dvj = f(xj, xi, mj)
    return np.array([vi, vj, dvi, dvj])

# ----------------------- SETUP FOR CALCULATIONS & ANALYTICAL SOLUTIONS -----------------------
tmax = 20
dt_arr = np.array([0.001, 0.005, 0.1, 0.25, 0.5, 0.75, 1])

t_arr = []
for dt in dt_arr:
    t_arr.append(np.arange(start = 0, stop = tmax + dt, step = dt))
print(type(t_arr[0]))
# Calculate analytic solution
#x1_as, x2_as = coupledOscillator(m1, m2, x10, x20, v10, v20, k, t_arr)


# Dictionary of lists of arrays for the Symplectic Euler Model
euler_arr = {'x1': [np.array([x10]) for i in dt_arr],
             'v1': [np.array([v10]) for i in dt_arr],
             'x2': [np.array([x10]) for i in dt_arr],
             'v2': [np.array([v10]) for i in dt_arr],
             }

# Dictionary of lists of arrays for the RK4 Model
rk4_arr = {'x1': [np.array([x10]) for i in dt_arr],
             'v1': [np.array([v10]) for i in dt_arr],
             'x2': [np.array([x10]) for i in dt_arr],
             'v2': [np.array([v10]) for i in dt_arr],
             }

# Dictionary of lists of arrays for the analytic solution
an_arr = {'x1': [np.array([x10]) for i in dt_arr],
             'v1': [np.array([v10]) for i in dt_arr],
             'x2': [np.array([x10]) for i in dt_arr],
             'v2': [np.array([v10]) for i in dt_arr],
             }

#test = euler_arr['x1'][0]       # returned --> [1] as expected
#print(test)

# get results for multiple dt values
# for i, dt in enumerate(dt_arr):
#     for j in len(t_arr):
#         for k in range(len(t_arr[j])-1):
#             # Evaluate at the most recent ("nth") step
#             x1n = euler_arr['x1'][i][-1]
#             x2n = euler_arr['x2'][i][-1]
#             v1n = euler_arr['v1'][i][-1]
#             v2n = euler_arr['v2'][i][-1]
#             s1np1 = meth.Euler(f, v1n, x1n, dt, xi = x2n, m = m1)
#             s2np1 = meth.Euler(f, v2n, x2n, dt, xi = x1n, m = m2)
#             # extract xn+1
#             x1np1 = s1np1[0]
#             x2np1 = s2np1[0]
#             # extract vn+1
#             v1np1 = s1np1[1]
#             v2np1 = s2np1[1]
#             # append n+1th values
#             euler_arr['x1'][i] = np.append(euler_arr['x1'][i], x1np1)
#             euler_arr['x2'][i] = np.append(euler_arr['x2'][i], x2np1)
#             euler_arr['v1'][i] = np.append(euler_arr['v1'][i], v1np1)
#             euler_arr['v2'][i] = np.append(euler_arr['v2'][i], v2np1)
    

test = euler_arr['x1'][0]       # returned --> [1] as expected
#print(np.shape(test))
#print(test)

#fig, ax = plt.subplots()
#ax.plot(t_arr, euler_arr['x1'][0])

# ----------------------- ARRAY CALCULATIONS FOR FIGURES -----------------------
x1e = np.array([x10])
x2e = np.array([x20])
v1e = np.array([v10])
v2e = np.array([v20])

# for i in range(len(t_arr)-1):
#     # nth positions
#     x1n = x1e[-1]
#     x2n = x2e[-1]
#     # nth velocities
#     v1n = v1e[-1]
#     v2n = v2e[-1]
#     #n_1th state vectors
#     s1np1 = meth.Euler(f, v1n, x1n, dt, xi = x2n, m = m1)
#     s2np1 = meth.Euler(f, v2n, x2n, dt, xi = x1n, m = m2)
#     # extract xn+1
#     x1np1 = s1np1[0]
#     x2np1 = s2np1[0]
#     # extract vn+1
#     v1np1 = s1np1[1]
#     v2np1 = s2np1[1]
#     # append n+1th values
#     x1e = np.append(x1e, x1np1)
#     x2e = np.append(x2e, x2np1)
#     v1e = np.append(v1e, v1np1)
#     v2e = np.append(v2e, v2np1)

# x1r = np.array([x10])
# x2r = np.array([x20])
# v1r = np.array([v10])
# v2r = np.array([v20])

# for i in range(len(t_arr)-1):
#     x1n = x1r[-1]
#     x2n = x2r[-1]

#     v1n = v1r[-1]
#     v2n = v2r[-1]
#     # n+1th state vectors
#     s1np1, s2np1 = meth.RK4(f, m1, m2, v1n, v2n, x1n, x2n, dt)
#     # extract xn+1
#     x1np1 = s1np1[0]
#     x2np1 = s2np1[0]
#     # extract vn+1
#     v1np1 = s1np1[1]
#     v2np1 = s2np1[1]
#     # append n+1th values
#     x1r = np.append(x1r, x1np1)
#     x2r = np.append(x2r, x2np1)
#     v1r = np.append(v1r, v1np1)
#     v2r = np.append(v2r, v2np1)

# # ----------- SCIPY ARRAYS ---------------
# s0 = np.array([x10, x20, v10, v20])
# # Tuple input for SciPy RK4
# t_range = (t_arr[0], t_arr[-1])
# # Compare to SciPy
# scipy = solve_ivp(fNew, t_range, s0, method = 'RK45', t_eval=t_arr, args = (m1, m2))

# x1sp = scipy.y[0]
# x2sp = scipy.y[1]
# v1sp = scipy.y[2]
# v2sp = scipy.y[3]

# # ----------------------------- ENERGY ---------------------------
# # Euler
# Ue, Te, Ee = coupledEnergy(m1, m2, v1e, v2e, x1e, x2e)
# # RK4
# Ur, Tr, Er = coupledEnergy(m1, m2, v1r, v2r, x1r, x2r)
# # Analytic
# Uas, Tas, Eas = coupledEnergy(m1, m2, v1r, v2r, x1_as, x2_as)

# # ----------------------------- ERRORS ---------------------------
# # Expected Errors
# # Fill all of the error arrays with 0s since there shouldn't be errors at the initial time

# # Actual Numerical Errors
# # Global Truncation Errors: Euler
# gerr1_e = np.abs(x1_as - x1e)
# gerr2_e = np.abs(x2_as - x2e)
# # Local Truncation Errors: Euler
# lerr1_e = np.array([0])
# lerr2_e = np.array([0])
# for n in range(len(t_arr) - 1):
#     # nth positions
#     x1n = x1_as[n]
#     x2n = x2_as[n]
#     # nth velocities
#     v1n = v1r[n]
#     v2n = v2r[n]
#     #n_1th state vectors
#     s1np1 = meth.Euler(f, v1n, x1n, dt, xi = x2n, m = m1)
#     s2np1 = meth.Euler(f, v2n, x2n, dt, xi = x1n, m = m2)
#     # extract xn+1
#     x1np1 = s1np1[0]
#     x2np1 = s2np1[0]
#     # extract vn+1
#     v1np1 = s1np1[1]
#     v2np1 = s2np1[1]
#     # append n+1th values
#     lerr1_e = np.append(lerr1_e, np.abs(x1_as[n+1] - x1np1))
#     lerr2_e = np.append(lerr2_e, np.abs(x2_as[n+1] - x2np1))
# print(len(lerr1_e))
# print(len(lerr2_e))
# # THIS ONE SEEMS TO WORK TOO HOORAY

# # Global Truncation Errors: RK4
# gerr1_r = np.abs(x1_as - x1r)
# gerr2_r = np.abs(x2_as - x2r)
# # Local Truncation Errors: RK4
# lerr1_r = np.array([0])
# lerr2_r = np.array([0])

# for n in range(len(t_arr) - 1):
#     x1n = x1_as[n]
#     x2n = x2_as[n]

#     #NOTE: NOT CORRECT ANALYTIC VELOCITIES! Already noted in overleaf doc
#     v1n = v1r[n]
#     v2n = v2r[n]

#     s1np1, s2np1 = meth.RK4(f, m1, m2, v1n, v2n, x1n, x2n, dt)
#     # extract xn+1
#     x1np1 = s1np1[0]
#     x2np1 = s2np1[0]
#     # extract vn+1
#     v1np1 = s1np1[1]
#     v2np1 = s2np1[1]
#     lerr1_r = np.append(lerr1_r, np.abs(x1_as[n+1] - x1np1))
#     lerr2_r = np.append(lerr2_r, np.abs(x2_as[n+1] - x2np1))

# # I copied and pasted these to get my error lists
# print('Euler: ', [np.mean(lerr1_e), np.mean(lerr2_e), np.mean(gerr1_e), np.mean(gerr2_e)])
# print('RK4: ', [np.mean(lerr1_r), np.mean(lerr2_r), np.mean(gerr1_r), np.mean(gerr2_r)])

# I THINK IT WORKS YIPPEE

# -------------------------PLOTS-----------------------------------
#--------------- MY SOLVERS VS ANALYTIC ----------------------
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,8))
# fig.suptitle(rf'Numerical Solvers vs Analytic Solutions: $\Delta t = ${dt}')
# ax1.set_title(rf'$m_1$ = m')
# ax2.set_title(rf'$m_2$ = {m2}m')
# ax1.plot(t_arr, x1e, label = r'$x_1(t)$ Euler', linestyle = ':')
# ax2.plot(t_arr, x2e, label = r'$x_2(t)$ Euler', linestyle = ':')

# ax1.plot(t_arr, x1r, label = r'$x_1(t)$ RK4', linestyle = ':')
# ax2.plot(t_arr, x2r, label = r'$x_2(t)$ RK4', linestyle = ':')

# ax1.plot(t_arr, x1_as, label = r'$x_1(t)$ Analytic Solution', linestyle = ':')
# ax2.plot(t_arr, x2_as, label = r'$x_2(t)$ Analytic Solution', linestyle = ':')

# ax1.legend()
# ax2.legend()
# plt.tight_layout()
# plt.savefig('solvers_v_an.png')

# fig, ax = plt.subplots(figsize = (8,5))
# fig.suptitle(rf'Numerical Solvers vs Analytic Solutions: $\Delta t = ${dt}')
# ax.set_title(r'$m_1=m_2=m$')
# ax.plot(t_arr, x1e, label = r'$x_1(t)$ Euler', linestyle = '-')
# ax.plot(t_arr, x2e, label = r'$x_2(t)$ Euler', linestyle = '--')

# ax.plot(t_arr, x1r, label = r'$x_1(t)$ RK4', linestyle = '-')
# ax.plot(t_arr, x2r, label = r'$x_2(t)$ RK4', linestyle = '--')

# ax.legend()
# ax.legend()
# plt.tight_layout()
#plt.savefig('solvers_v_an.png')

# # ---------------- SCIPY SOLVER VS ANALYTIC SOLUTION ----------------
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,8))
# ax1.set_title(rf'$m_1$ = $m$')
# ax2.set_title(rf'$m_2$ = {m2}$m$')
# fig.suptitle(rf'SciPy Solvers vs Analytic Solutions: $\Delta t = ${dt}')

# ax1.plot(t_arr, x1_as, label = r'$x_1(t)$ Analytic Solution', color = 'C2')
# ax2.plot(t_arr, x2_as, label = r'$x_2(t)$ Analytic Solution', color = 'C2')

# #---------- SCIPY solve_ivp SOLUTIONS ------------
# ax1.plot(t_arr, x1sp, label = r'$x_1(t)$ SciPy RK45', color = 'C3', linestyle = '--')
# ax2.plot(t_arr, x2sp, label = r'$x_2(t)$ SciPy RK45', color = 'C3', linestyle = '--')

# ax1.legend()
# ax2.legend()
# plt.tight_layout()
# plt.savefig('an_v_scipy.png')

# # # ---------------- SCIPY SOLVER VS MY RK4 CODE ----------------
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,8))
# ax1.set_title(rf'$m_1$ = m')
# ax2.set_title(rf'$m_2$ = {m2}m')
# fig.suptitle(rf'Numerical Solvers vs SciPy Solvers: $\Delta t = ${dt}')

# ax1.plot(t_arr, x1r, label = r'$x_1(t)$ RK4', color = 'C1', linestyle = '--')
# ax2.plot(t_arr, x2r, label = r'$x_2(t)$ RK4', color = 'C1', linestyle = '-')


# ax1.plot(t_arr, x1sp, label = r'$x_1(t)$ solve_ivp RK45', color = 'C3', linestyle = '--')
# ax2.plot(t_arr, x2sp, label = r'$x_2(t)$ solve_ivp RK45', color = 'C3', linestyle = '--')

# ax1.legend()
# ax2.legend()
# plt.tight_layout()
# plt.savefig('rk_v_scipy.png')

# #---------- II: ENERGY CONSERVATION ------------

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (8,12))
# fig.suptitle(rf'Energy conservation: $\Delta t = ${dt}, $m_2=${m2/m1}$m_1$')

# ax1.set_title('Symplectic Euler')
# ax1.plot(t_arr, Ue, label = r'$U(t)$')
# ax1.plot(t_arr, Te, label = r'$T(t)$')
# ax1.plot(t_arr, Ee, label = r'$E(t)$')

# ax2.set_title('Runge-Kutta 4')
# ax2.plot(t_arr, Ur, label = r'$U(t)$')
# ax2.plot(t_arr, Tr, label = r'$T(t)$')
# ax2.plot(t_arr, Er, label = r'$E(t)$')

# ax3.set_title('Analytic Solution')
# ax3.plot(t_arr, Uas, label = r'$U(t)$')
# ax3.plot(t_arr, Tas, label = r'$T(t)$')
# ax3.plot(t_arr, Eas, label = r'$E(t)$')

# ax1.legend()
# ax2.legend()
# ax3.legend()

# plt.tight_layout()

# plt.savefig('energy_cons_dt0p6.png')

# # ---------------- ERROR PROPAGATION ----------------
# dt_arr = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
# O_1 = dt_arr
# O_2 = dt_arr ** 2
# O_4 = dt_arr ** 4
# O_5 = dt_arr ** 5

# lerr1_e_avg = np.array([3.6369e-05, 9.0811e-04, 3.6294e-03, 8.1709e-03, 1.4514e-02, 2.2608e-02,
#                         3.2578e-02, 4.4229e-02, 5.7531e-02, 7.2090e-02, 8.9394e-02])
# lerr2_e_avg = np.array([1.4091e-05, 3.5070e-04, 1.3957e-03, 3.1354e-03, 5.5260e-03, 8.5943e-03,
#                         1.2355e-02, 1.6851e-02, 2.1685e-02, 2.7520e-02, 3.3582e-02])
# gerr1_e_avg = np.array([2.1538e-03, 1.1371e-02, 2.4296e-02, 3.8733e-02, 5.4985e-02, 7.3061e-02,
#                         9.3019e-02, 1.1440e-01, 1.3773e-01, 1.6344e-01, 1.9179e-01])
# gerr2_e_avg = np.array([1.2245e-03, 6.0758e-03, 1.2043e-02, 1.7967e-02, 2.3675e-02, 2.9345e-02,
#                         3.5065e-02, 4.0914e-02, 4.5829e-02, 5.1773e-02, 5.6375e-02])

# lerr1_r_avg = np.array([2.4987e-11, 7.7724e-08, 2.4722e-06, 1.8807e-05, 7.7992e-05, 2.3522e-04,
#                         5.8242e-04, 1.2590e-03, 2.3702e-03, 4.2588e-03, 6.9235e-03])
# lerr2_r_avg = np.array([1.3474e-12, 4.1889e-09, 1.3324e-07, 1.0131e-06, 4.1975e-06, 1.2665e-05,
#                         3.1356e-05, 6.7820e-05, 1.2730e-04, 2.2871e-04, 3.7245e-04])
# gerr1_r_avg = np.array([1.7727e-09, 1.1089e-06, 1.7764e-05, 8.9834e-05, 2.8467e-04, 6.9467e-04,
#                         1.4418e-03, 2.6477e-03, 4.5522e-03, 7.2094e-03, 1.0997e-02])
# gerr2_r_avg = np.array([9.5651e-11, 5.9831e-08, 9.5831e-07, 4.8477e-06, 1.5362e-05, 3.7563e-05,
#                         7.7765e-05, 1.4287e-04, 2.4583e-04, 3.9051e-04, 5.9955e-04])

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12,8))
# ax1.set_title('Symplectic Euler: Local Error')
# ax1.loglog(dt_arr, lerr1_e_avg, label = 'Local Error for Mass 1')
# ax1.loglog(dt_arr, lerr2_e_avg, label = 'Local Error for Mass 2')
# ax1.loglog(dt_arr, O_2, label = r'$\mathcal{O}(\Delta t^2)$')

# ax2.set_title('Symplectic Euler: Global Error')
# ax2.loglog(dt_arr, gerr1_e_avg, label = 'Global Error for Mass 1')
# ax2.loglog(dt_arr, gerr2_e_avg, label = 'Global Error for Mass 2')
# ax2.loglog(dt_arr, O_1, label = r'$\mathcal{O}(\Delta t)$')

# ax3.set_title('Runge-Kutta 4: Local Error')
# ax3.loglog(dt_arr, lerr1_r_avg, label = 'Local Error for Mass 1')
# ax3.loglog(dt_arr, lerr2_r_avg, label = 'Local Error for Mass 2')
# ax3.loglog(dt_arr, O_5, label = r'$\mathcal{O}(\Delta t^5)$')

# ax4.set_title('Runge-Kutta 4: Global Error')
# ax4.loglog(dt_arr, gerr1_r_avg, label = 'Global Error for Mass 1')
# ax4.loglog(dt_arr, gerr2_r_avg, label = 'Global Error for Mass 2')
# ax4.loglog(dt_arr, O_4, label = r'$\mathcal{O}(\Delta t^4)$')

# ax1.legend()
# ax2.legend()
# ax3.legend()
# ax4.legend()

# plt.tight_layout()
# plt.savefig('ODE_err.png')
plt.show()
