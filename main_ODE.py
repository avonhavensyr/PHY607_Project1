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

t_lst = []
for dt in dt_arr:
    t_lst.append(np.arange(start = 0, stop = tmax + dt, step = dt))
print(type(t_lst[0]))

an_dict = {'x1': [],
             'x2': [],
             'v1': [],
             'v2': [],
             }

# Dictionary of lists of arrays for the Symplectic Euler Model
euler_dict = {'x1': [np.array([x10]) for i in dt_arr],
             'v1': [np.array([v10]) for i in dt_arr],
             'x2': [np.array([x10]) for i in dt_arr],
             'v2': [np.array([v10]) for i in dt_arr],
             }

# Dictionary of lists of arrays for the RK4 Model
rk4_dict = {'x1': [np.array([x10]) for i in dt_arr],
             'v1': [np.array([v10]) for i in dt_arr],
             'x2': [np.array([x10]) for i in dt_arr],
             'v2': [np.array([v10]) for i in dt_arr],
             }

# Dictionary of lists of arrays for the SciPy solver
sp_dict = {'x1': [],
             'v1': [],
             'x2': [],
             'v2': [],
             }


    

# ----------------------- CALCULATIONS FOR FIGURES -----------------------

# -------------------- Analytic Solution --------------------
for i, dt in enumerate(dt_arr):
    x1, x2 = coupledOscillator(m1, m2, x10, x20, v10, v20, k, t_lst[i])
    an_dict['x1'].append(x1)
    an_dict['x2'].append(x2)

# -------------------- Symplectic Euler --------------------
for i, dt in enumerate(dt_arr):
    for j in range(len(t_lst[i])-1):
        # Evaluate at the most recent ("nth") step
        x1n = euler_dict['x1'][i][-1]
        x2n = euler_dict['x2'][i][-1]
        v1n = euler_dict['v1'][i][-1]
        v2n = euler_dict['v2'][i][-1]
        s1np1 = meth.Euler(f, v1n, x1n, dt, xi = x2n, m = m1)
        s2np1 = meth.Euler(f, v2n, x2n, dt, xi = x1n, m = m2)
        # extract xn+1
        x1np1 = s1np1[0]
        x2np1 = s2np1[0]
        # extract vn+1
        v1np1 = s1np1[1]
        v2np1 = s2np1[1]
        # append n+1th values
        euler_dict['x1'][i] = np.append(euler_dict['x1'][i], x1np1)
        euler_dict['x2'][i] = np.append(euler_dict['x2'][i], x2np1)
        euler_dict['v1'][i] = np.append(euler_dict['v1'][i], v1np1)
        euler_dict['v2'][i] = np.append(euler_dict['v2'][i], v2np1)

# -------------------- 4th-Order Runge Kutta --------------------
for i, dt in enumerate(dt_arr):
    for j in range(len(t_lst[i])-1):
        x1n = rk4_dict['x1'][i][-1]
        x2n = rk4_dict['x2'][i][-1]
        v1n = rk4_dict['v1'][i][-1]
        v2n = rk4_dict['v2'][i][-1]

        # n+1th state vectors
        s1np1, s2np1 = meth.RK4(f, m1, m2, v1n, v2n, x1n, x2n, dt)
        # extract xn+1
        x1np1 = s1np1[0]
        x2np1 = s2np1[0]
        # extract vn+1
        v1np1 = s1np1[1]
        v2np1 = s2np1[1]
        # append n+1th values
        rk4_dict['x1'][i] = np.append(rk4_dict['x1'][i], x1np1)
        rk4_dict['x2'][i] = np.append(rk4_dict['x2'][i], x2np1)
        rk4_dict['v1'][i] = np.append(rk4_dict['v1'][i], v1np1)
        rk4_dict['v2'][i] = np.append(rk4_dict['v2'][i], v2np1)

# -------------------- SCIPY RK45 Method --------------------
s0 = np.array([x10, x20, v10, v20])
# Tuple input for SciPy RK4

for i, t_arr in enumerate(t_lst):
    t_range = (t_lst[i][0], t_lst[i][-1])
    # Compare to SciPy
    scipy = solve_ivp(fNew, t_range, s0, method = 'RK45', t_eval=t_arr, args = (m1, m2))

    sp_dict['x1'].append(scipy.y[0])
    sp_dict['x2'].append(scipy.y[1])
    sp_dict['v1'].append(scipy.y[2])
    sp_dict['v2'].append(scipy.y[3])

# Indices for the small and large dt values
s_idx = 0
l_idx = -2

# Variables for cleaner plotting
x1e = euler_dict['x1']
x2e = euler_dict['x2']
v1e = euler_dict['v1']
v2e = euler_dict['v2']

x1r = rk4_dict['x1']
x2r = rk4_dict['x2']
v1r = rk4_dict['v1']
v2r = rk4_dict['v2']

x1a = an_dict['x1']
x2a = an_dict['x2']
v1a = an_dict['v1']
v2a = an_dict['v2']

x1s = sp_dict['x1']
x2s = sp_dict['x2']
v1s = sp_dict['v1']
v2s = sp_dict['v2']

# ----------------------------- ENERGY ---------------------------
# Euler
sUe, sTe, sEe = coupledEnergy(m1, m2, v1e[s_idx], v2e[s_idx], x1e[s_idx], x2e[s_idx])
# RK4
sUr, sTr, sEr = coupledEnergy(m1, m2, v1r[s_idx], v2r[s_idx], x1r[s_idx], x2r[s_idx])
# Analytic
sUas, sTas, sEas = coupledEnergy(m1, m2, v1r[s_idx], v2r[s_idx], x1a[s_idx], x2a[s_idx])
# Euler
lUe, lTe, lEe = coupledEnergy(m1, m2, v1e[l_idx], v2e[l_idx], x1e[l_idx], x2e[l_idx])
# RK4
lUr, lTr, lEr = coupledEnergy(m1, m2, v1r[l_idx], v2r[l_idx], x1r[l_idx], x2r[l_idx])
# Analytic
lUas, lTas, lEas = coupledEnergy(m1, m2, v1r[l_idx], v2r[l_idx], x1a[l_idx], x2a[l_idx])

# ----------------------------- ERRORS ---------------------------
# Expected Errors
# Fill all of the error arrays with 0s since there shouldn't be errors at the initial time

# Global Truncation Errors
gerr1_e = []
gerr2_e = []
gerr1_r = []
gerr2_r = []

for i, dt in enumerate(dt_arr):
    # SI Euler
    gerr1_e.append(np.abs(x1a[i] - x1e[i]))
    gerr2_e.append(np.abs(x2a[i] - x2e[i]))
    # RK4
    gerr1_r.append(np.abs(x1a[i] - x1r[i]))
    gerr2_r.append(np.abs(x2a[i] - x2r[i]))
# Local Truncation Errors
# SI Euler
lerr1_e = [np.array([0]) for dt in dt_arr]
lerr2_e = [np.array([0]) for dt in dt_arr]
# RK4
lerr1_r = [np.array([0]) for dt in dt_arr]
lerr2_r = [np.array([0]) for dt in dt_arr]

for i, dt in enumerate(dt_arr):
    for n in range(len(t_lst[i]) - 1):
        # nth positions
        x1n = x1a[i][n]
        x2n = x2a[i][n]
        # NOTE: NOT CORRECT ANALYTIC VELOCITIES! Already noted in overleaf doc
        # nth velocities
        v1n = v1r[i][n]
        v2n = v2r[i][n]
        # EULER
        #n_1th state vectors
        e_s1np1 = meth.Euler(f, v1n, x1n, dt, xi = x2n, m = m1)
        e_s2np1 = meth.Euler(f, v2n, x2n, dt, xi = x1n, m = m2)
        # extract xn+1
        x1e_np1 = e_s1np1[0]
        x2e_np1 = e_s2np1[0]
        # extract vn+1
        v1e_np1 = e_s1np1[1]
        v2e_np1 = e_s2np1[1]
        # append n+1th values
        lerr1_e[i] = np.append(lerr1_e[i], np.abs(x1a[i][n+1] - x1e_np1))
        lerr2_e[i] = np.append(lerr2_e[i], np.abs(x2a[i][n+1] - x2e_np1))

        # RK4
        r_s1np1, r_s2np1 = meth.RK4(f, m1, m2, v1n, v2n, x1n, x2n, dt)
        # extract xn+1
        x1r_np1 = r_s1np1[0]
        x2r_np1 = r_s2np1[0]
        # extract vn+1
        v1r_np1 = r_s1np1[1]
        v2r_np1 = r_s2np1[1]
        lerr1_r[i] = np.append(lerr1_r[i], np.abs(x1a[i][n+1] - x1r_np1))
        lerr2_r[i] = np.append(lerr2_r[i], np.abs(x2a[i][n+1] - x2r_np1))

# Average local errors per dt
avg_lerr1_e = [np.mean(i) for i in lerr1_e]
avg_lerr2_e = [np.mean(i) for i in lerr2_e]
avg_lerr1_r = [np.mean(i) for i in lerr1_r]
avg_lerr2_r = [np.mean(i) for i in lerr2_r]

avg_gerr1_e = [np.mean(i) for i in gerr1_e]
avg_gerr2_e = [np.mean(i) for i in gerr2_e]
avg_gerr1_r = [np.mean(i) for i in gerr1_r]
avg_gerr2_r = [np.mean(i) for i in gerr2_r]

print(gerr1_e[0])
# THIS ONE SEEMS TO WORK TOO HOORAY

# Global Truncation Errors: RK4
# Local Truncation Errors: RK4









# ----------------------- PLOTS-----------------------

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8), sharex = True)

ax1.set_title(rf'Mass 1: $\Delta t=${dt_arr[s_idx]}')
ax1.plot(t_lst[s_idx], x1a[s_idx], label = 'Analytic', ls = '-')
ax1.plot(t_lst[s_idx], x1e[s_idx], label = 'Euler', ls = '--')
ax1.plot(t_lst[s_idx], x1r[s_idx], label = 'RK4', ls = '-')
ax1.plot(t_lst[s_idx], x1s[s_idx], label = 'SciPy RK45', ls = '--')

ax2.set_title(rf'Mass 2: $\Delta t=${dt_arr[s_idx]}')
ax2.plot(t_lst[s_idx], x2a[s_idx], label = 'Analytic', ls = '-')
ax2.plot(t_lst[s_idx], x2e[s_idx], label = 'Euler', ls = '--')
ax2.plot(t_lst[s_idx], x2r[s_idx], label = 'RK4', ls = '-')
ax2.plot(t_lst[s_idx], x2s[s_idx], label = 'SciPy RK45', ls = '--')

ax3.set_title(rf'Mass 1: $\Delta t=${dt_arr[l_idx]}')
ax3.plot(t_lst[l_idx], x1a[l_idx], label = 'Analytic', ls = '-')
ax3.plot(t_lst[l_idx], x1e[l_idx], label = 'Euler', ls = '--')
ax3.plot(t_lst[l_idx], x1r[l_idx], label = 'RK4', ls = '-')
ax3.plot(t_lst[l_idx], x1s[l_idx], label = 'SciPy RK45', ls = '--')

ax4.set_title(rf'Mass 2: $\Delta t=${dt_arr[l_idx]}')
ax4.plot(t_lst[l_idx], x2a[l_idx], label = 'Analytic', ls = '-')
ax4.plot(t_lst[l_idx], x2e[l_idx], label = 'Euler', ls = '--')
ax4.plot(t_lst[l_idx], x2r[l_idx], label = 'RK4', ls = '-')
ax4.plot(t_lst[l_idx], x2s[l_idx], label = 'SciPy RK45', ls = '--')

ax1.legend(loc = 'lower left')
ax2.legend(loc = 'lower left')
ax3.legend(loc = 'lower left')
ax4.legend(loc = 'lower left')

fig.supxlabel('t')

plt.tight_layout()
#plt.savefig('NEWode_comparisons.png')
plt.savefig('NEWlim_case1.png')


#---------- II: Energy Conservation ------------
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(8, 10), sharex = True)

ax1.set_title(rf'Symplectic Euler: $\Delta t=${dt_arr[s_idx]}')
ax1.plot(t_lst[s_idx], sUe, label = r'$U(t)$')
ax1.plot(t_lst[s_idx], sTe, label = r'$T(t)$')
ax1.plot(t_lst[s_idx], sEe, label = r'$E(t)$')

ax2.set_title(rf'Symplectic Euler: $\Delta t=${dt_arr[l_idx]}')
ax2.plot(t_lst[l_idx], lUe, label = r'$U(t)$')
ax2.plot(t_lst[l_idx], lTe, label = r'$T(t)$')
ax2.plot(t_lst[l_idx], lEe, label = r'$E(t)$')

# Row 2: Runge-Kutta 4
ax3.set_title(rf'Runge-Kutta 4: $\Delta t=${dt_arr[s_idx]}')
ax3.plot(t_lst[s_idx], sUr, label = r'$U(t)$')
ax3.plot(t_lst[s_idx], sTr, label = r'$T(t)$')
ax3.plot(t_lst[s_idx], sEr, label = r'$E(t)$')

ax4.set_title(rf'Runge-Kutta 4: $\Delta t=${dt_arr[l_idx]}')
ax4.plot(t_lst[l_idx], lUr, label = r'$U(t)$')
ax4.plot(t_lst[l_idx], lTr, label = r'$T(t)$')
ax4.plot(t_lst[l_idx], lEr, label = r'$E(t)$')

# Row 3: Analytic
ax5.set_title(rf'Analytic: $\Delta t=${dt_arr[s_idx]}')
ax5.plot(t_lst[s_idx], sUas, label = r'$U(t)$')
ax5.plot(t_lst[s_idx], sTas, label = r'$T(t)$')
ax5.plot(t_lst[s_idx], sEas, label = r'$E(t)$')

ax6.set_title(rf'Analytic: $\Delta t=${dt_arr[l_idx]}')
ax6.plot(t_lst[l_idx], lUas, label = r'$U(t)$')
ax6.plot(t_lst[l_idx], lTas, label = r'$T(t)$')
ax6.plot(t_lst[l_idx], lEas, label = r'$E(t)$')

fig.supxlabel('t')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
ax6.legend()

plt.tight_layout()

plt.savefig('NEWode_energy_cons.png')

# ---------------- ERROR PROPAGATION ----------------
O_1 = dt_arr
O_2 = dt_arr ** 2
O_4 = dt_arr ** 4
O_5 = dt_arr ** 5

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Mass 1')
ax1.set_title('Symplectic Euler: Average Local & Global Error')
ax2.set_title('Symplectic Euler: Local Error')
ax3.set_title('RK4: Average Local & Global Error')
ax4.set_title('RK4: Local Error')

ax1.loglog(dt_arr, O_1, label = r'$\mathcal{O}(\Delta t)$',color = 'C0',ls='--')
ax1.loglog(dt_arr, O_2, label = r'$\mathcal{O}(\Delta t^2)$',color = 'C1',ls='--')
ax3.loglog(dt_arr, O_4, label = r'$\mathcal{O}(\Delta t^4)$',color = 'C0',ls='--')
ax3.loglog(dt_arr, O_5, label = r'$\mathcal{O}(\Delta t^5)$',color = 'C1',ls='--')

ax1.loglog(dt_arr, avg_gerr1_e, label = 'Global Error')
ax1.loglog(dt_arr, avg_lerr1_e, label = 'Average Local Error')
ax3.loglog(dt_arr, avg_gerr1_r, label = 'Global Error')
ax3.loglog(dt_arr, avg_lerr1_r, label = 'Average Local Error')

for i, dt in enumerate(dt_arr):
    ax2.loglog(t_lst[i], lerr1_e[i], label = rf'$\Delta t=${dt}')
    ax4.loglog(t_lst[i], lerr1_r[i], label = rf'$\Delta t=${dt}')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

ax1.set_xlabel(r'$\Delta t$')
ax3.set_xlabel(r'$\Delta t$')
ax1.set_ylabel(r'Error')
ax3.set_ylabel(r'Error')

ax2.set_xlabel(r't')
ax4.set_xlabel(r't')
ax2.set_ylabel(r'Error')
ax4.set_ylabel(r'Error')

plt.tight_layout()
plt.savefig('ODE_err1.png')
plt.show()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Mass 2')
ax1.set_title('Symplectic Euler: Average Local & Global Error')
ax2.set_title('Symplectic Euler: Local Error')
ax3.set_title('RK4: Average Local & Global Error')
ax4.set_title('RK4: Local Error')

ax1.loglog(dt_arr, O_1, label = r'$\mathcal{O}(\Delta t)$',color = 'C0',ls='--')
ax1.loglog(dt_arr, O_2, label = r'$\mathcal{O}(\Delta t^2)$',color = 'C1',ls='--')
ax3.loglog(dt_arr, O_4, label = r'$\mathcal{O}(\Delta t^4)$',color = 'C0',ls='--')
ax3.loglog(dt_arr, O_5, label = r'$\mathcal{O}(\Delta t^5)$',color = 'C1',ls='--')

ax1.loglog(dt_arr, avg_gerr2_e, label = 'Global Error')
ax1.loglog(dt_arr, avg_lerr2_e, label = 'Average Local Error')
ax3.loglog(dt_arr, avg_gerr2_r, label = 'Global Error')
ax3.loglog(dt_arr, avg_lerr2_r, label = 'Average Local Error')

for i, dt in enumerate(dt_arr):
    ax2.loglog(t_lst[i], lerr2_e[i], label = rf'$\Delta t=${dt}')
    ax4.loglog(t_lst[i], lerr2_r[i], label = rf'$\Delta t=${dt}')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

ax1.set_xlabel(r'$\Delta t$')
ax3.set_xlabel(r'$\Delta t$')
ax1.set_ylabel(r'Error')
ax3.set_ylabel(r'Error')

ax2.set_xlabel(r't')
ax4.set_xlabel(r't')
ax2.set_ylabel(r'Error')
ax4.set_ylabel(r'Error')

plt.tight_layout()
plt.savefig('ODE_err2.png')
plt.show()
