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
dt = 0.2

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
t_arr = np.arange(start = 0, stop = tmax + dt, step = dt)

# Calculate analytic solution
x1_as, x2_as = coupledOscillator(m1, m2, x10, x20, v10, v20, k, t_arr)


# ----------------------- ARRAY CALCULATIONS FOR FIGURES -----------------------
x1e = np.array([x10])
x2e = np.array([x20])
v1e = np.array([v10])
v2e = np.array([v20])

for i in range(len(t_arr)-1):
    # nth positions
    x1n = x1e[-1]
    x2n = x2e[-1]
    # nth velocities
    v1n = v1e[-1]
    v2n = v2e[-1]
    #n_1th state vectors
    s1np1 = meth.Euler(f, v1n, x1n, dt, xi = x2n, m = m1)
    s2np1 = meth.Euler(f, v2n, x2n, dt, xi = x1n, m = m2)
    # extract xn+1
    x1np1 = s1np1[0]
    x2np1 = s2np1[0]
    # extract vn+1
    v1np1 = s1np1[1]
    v2np1 = s2np1[1]
    # append n+1th values
    x1e = np.append(x1e, x1np1)
    x2e = np.append(x2e, x2np1)
    v1e = np.append(v1e, v1np1)
    v2e = np.append(v2e, v2np1)

x1r = np.array([x10])
x2r = np.array([x20])
v1r = np.array([v10])
v2r = np.array([v20])

for i in range(len(t_arr)-1):
    x1n = x1r[-1]
    x2n = x2r[-1]

    v1n = v1r[-1]
    v2n = v2r[-1]
    # n+1th state vectors
    s1np1, s2np1 = meth.RK4(f, m1, m2, v1n, v2n, x1n, x2n, dt)
    # extract xn+1
    x1np1 = s1np1[0]
    x2np1 = s2np1[0]
    # extract vn+1
    v1np1 = s1np1[1]
    v2np1 = s2np1[1]
    # append n+1th values
    x1r = np.append(x1r, x1np1)
    x2r = np.append(x2r, x2np1)
    v1r = np.append(v1r, v1np1)
    v2r = np.append(v2r, v2np1)

# ----------- SCIPY ARRAYS ---------------
s0 = np.array([x10, x20, v10, v20])
# Tuple input for SciPy RK4
t_range = (t_arr[0], t_arr[-1])
# Compare to SciPy
scipy = solve_ivp(fNew, t_range, s0, method = 'RK45', t_eval=t_arr, args = (m1, m2))

x1sp = scipy.y[0]
x2sp = scipy.y[1]
v1sp = scipy.y[2]
v2sp = scipy.y[3]

# ----------------------------- ENERGY ---------------------------
# Euler
Ue, Te, Ee = coupledEnergy(m1, m2, v1e, v2e, x1e, x2e)
# RK4
Ur, Tr, Er = coupledEnergy(m1, m2, v1r, v2r, x1r, x2r)
# Analytic
Uas, Tas, Eas = coupledEnergy(m1, m2, v1r, v2r, x1_as, x2_as)

# ----------------------------- ERRORS ---------------------------
# Expected Errors
lerr1_e_th = np.array([0])
lerr2_e_th = np.array([0])

gerr1_e_th = np.array([0])
gerr2_e_th = np.array([0])

lerr1_r_th = np.array([0])
lerr2_r_th = np.array([0])

gerr1_r_th = np.array([0])
gerr2_r_th = np.array([0])

# Global Truncation Errors: Euler
gerr1_e = np.abs(x1_as - x1e)
gerr2_e = np.abs(x2_as - x2e)
# Local Truncation Errors: Euler
lerr1_e = np.array([0])
lerr2_e = np.array([0])
for n in range(len(t_arr) - 1):
    # nth positions
    x1n = x1_as[n]
    x2n = x2_as[n]
    # nth velocities
    v1n = v1r[n]
    v2n = v2r[n]
    #n_1th state vectors
    s1np1 = meth.Euler(f, v1n, x1n, dt, xi = x2n, m = m1)
    s2np1 = meth.Euler(f, v2n, x2n, dt, xi = x1n, m = m2)
    # extract xn+1
    x1np1 = s1np1[0]
    x2np1 = s2np1[0]
    # extract vn+1
    v1np1 = s1np1[1]
    v2np1 = s2np1[1]
    # append n+1th values
    lerr1_e = np.append(lerr1_e, np.abs(x1_as[n+1] - x1np1))
    lerr2_e = np.append(lerr2_e, np.abs(x2_as[n+1] - x2np1))
print(len(lerr1_e))
print(len(lerr2_e))
# THIS ONE SEEMS TO WORK TOO HOORAY

# Global Truncation Errors: RK4
gerr1_r = np.abs(x1_as - x1r)
gerr2_r = np.abs(x2_as - x2r)
# Local Truncation Errors: RK4
lerr1_r = np.array([0])
lerr2_r = np.array([0])
for n in range(len(t_arr) - 1):
    x1n = x1_as[n]
    x2n = x2_as[n]

    #NOTE: NOT CORRECT ANALYTIC VELOCITIES! Already noted in overleaf doc
    v1n = v1r[n]
    v2n = v2r[n]

    s1np1, s2np1 = meth.RK4(f, m1, m2, v1n, v2n, x1n, x2n, dt)
    # extract xn+1
    x1np1 = s1np1[0]
    x2np1 = s2np1[0]
    # extract vn+1
    v1np1 = s1np1[1]
    v2np1 = s2np1[1]
    lerr1_r = np.append(lerr1_r, np.abs(x1_as[n+1] - x1np1))
    lerr2_r = np.append(lerr2_r, np.abs(x2_as[n+1] - x2np1))

print(f'Average Local Error for Mass 1 (Euler): {lerr1_e}')
print(f'Average Local Error for Mass 2 (Euler): {lerr2_e}')

print(f'Average Global Error for Mass 1 (Euler): {gerr1_e}')
print(f'Average Global Error for Mass 2 (Euler): {gerr2_e}')

print(f'Average Local Error for Mass 1 (RK4): {lerr1_r}')
print(f'Average Local Error for Mass 2 (RK4): {lerr2_r}')

print(f'Average Global Error for Mass 1(RK4): {gerr1_r}')
print(f'Average Global Error for Mass 2(RK4): {gerr2_r}')
# I THINK IT WORKS YIPPEE
print
# -------------------------PLOTS-----------------------------------
#--------------- MY SOLVERS VS ANALYTIC ----------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,8))
ax1.plot(t_arr, x1e, label = r'$x_1(t)$ Euler')
ax2.plot(t_arr, x2e, label = r'$x_2(t)$ Euler')

ax1.plot(t_arr, x1r, label = r'$x_1(t)$ RK4')
ax2.plot(t_arr, x2r, label = r'$x_2(t)$ RK4')

ax1.plot(t_arr, x1_as, label = r'$x_1(t)$ Analytic Solution')
ax2.plot(t_arr, x2_as, label = r'$x_2(t)$ Analytic Solution')

ax1.legend()
ax2.legend()
plt.tight_layout()
plt.savefig('solvers_v_an.png')

# ---------------- SCIPY SOLVER VS ANALYTIC SOLUTION ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,8))

ax1.plot(t_arr, x1_as, label = r'$x_1(t)$ Analytic Solution', color = 'C2')
ax2.plot(t_arr, x2_as, label = r'$x_2(t)$ Analytic Solution', color = 'C2')

#---------- SCIPY solve_ivp SOLUTIONS ------------
ax1.plot(t_arr, x1sp, label = r'$x_1(t)$ solve_ivp RK45', color = 'C3')
ax2.plot(t_arr, x2sp, label = r'$x_2(t)$ solve_ivp RK45', color = 'C3')

ax1.legend()
ax2.legend()
plt.tight_layout()
plt.savefig('an_v_scipy.png')

# ---------------- SCIPY SOLVER VS MY RK4 CODE ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,8))

ax1.plot(t_arr, x1r, label = r'$x_1(t)$ RK4', color = 'C1')
ax2.plot(t_arr, x2r, label = r'$x_2(t)$ RK4', color = 'C1')

# ------------------- TESTED PROPERTIES/LIMITING CASES -------------------
#---------- I: ENERGY CONSERVATION ------------
ax1.plot(t_arr, x1sp, label = r'$x_1(t)$ solve_ivp RK45', color = 'C3')
ax2.plot(t_arr, x2sp, label = r'$x_2(t)$ solve_ivp RK45', color = 'C3')

ax1.legend()
ax2.legend()
plt.tight_layout()
plt.savefig('rk_v_scipy.png')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (8,12))
ax1.set_title('Symplectic Euler')
ax1.plot(t_arr, Ue, label = r'$U(t)$')
ax1.plot(t_arr, Te, label = r'$T(t)$')
ax1.plot(t_arr, Ee, label = r'$E(t)$')

ax2.set_title('Runge-Kutta 4')
ax2.plot(t_arr, Ur, label = r'$U(t)$')
ax2.plot(t_arr, Tr, label = r'$T(t)$')
ax2.plot(t_arr, Er, label = r'$E(t)$')

ax3.set_title('Analytic Solution')
ax3.plot(t_arr, Uas, label = r'$U(t)$')
ax3.plot(t_arr, Tas, label = r'$T(t)$')
ax3.plot(t_arr, Eas, label = r'$E(t)$')

ax1.legend()
ax2.legend()
ax3.legend()

plt.tight_layout()

plt.savefig('energy_cons.png')

# ---------------- ERROR PROPAGATION ----------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (16,16))
ax1.set_title('Symplectic Euler: Local Error')
ax1.plot(t_arr, lerr1_e, label = 'Local Error for Mass 1')
ax1.plot(t_arr, lerr2_e, label = 'Local Error for Mass 2')

ax2.set_title('Symplectic Euler: Global Error')
ax2.plot(t_arr, gerr1_e, label = 'Global Error for Mass 1')
ax2.plot(t_arr, gerr2_e, label = 'Global Error for Mass 2')

ax3.set_title('Runge-Kutta 4: Local Error')
ax3.plot(t_arr, lerr1_r, label = 'Local Error for Mass 1')
ax3.plot(t_arr, lerr2_r, label = 'Local Error for Mass 2')

ax4.set_title('Runge-Kutta 4: Local Error')
ax4.plot(t_arr, gerr1_r, label = 'Global Error for Mass 1')
ax4.plot(t_arr, gerr2_r, label = 'Global Error for Mass 2')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.tight_layout()

plt.show()
