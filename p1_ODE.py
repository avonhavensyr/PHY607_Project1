import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import methods as meth

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

def coupledOscillator(m1, m2, x10, x20, v10, v20, k, t):
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

    return (x1, x2)

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
x1 = np.array([x10])
x2 = np.array([x20])
v1 = np.array([v10])
v2 = np.array([v20])

for i in range(len(t_arr)-1):
    # nth positions
    x1n = x1[-1]
    x2n = x2[-1]
    # nth velocities
    v1n = v1[-1]
    v2n = v2[-1]
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
    x1e = np.append(x1, x1np1)
    x2e = np.append(x2, x2np1)
    v1e = np.append(v1, v1np1)
    v2e = np.append(v2, v2np1)

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

# ----------- SCIPY COMPARISON ---------------
s0 = np.array([x10, x20, v10, v20])
# Tuple input for SciPy RK4
t_range = (t_arr[0], t_arr[-1])
# Compare to SciPy
scipy = solve_ivp(fNew, t_range, s0, method = 'RK45', t_eval=t_arr, args = (m1, m2))

x1sp = scipy.y[0]
x2sp = scipy.y[1]
v1sp = scipy.y[2]
v2sp = scipy.y[3]

#--------------- MY SOLVERS VS ANALYTIC ----------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,8))
ax1.plot(t_arr, x1e, label = f'Mass 1 Euler')
ax2.plot(t_arr, x2e, label = f'Mass 2 Euler')

ax1.plot(t_arr, x1r, label = f'Mass 1 RK4')
ax2.plot(t_arr, x2r, label = f'Mass 2 RK4')

ax1.plot(t_arr, x1_as, label = f'Mass 1 Analytic Solution')
ax2.plot(t_arr, x2_as, label = f'Mass 2 Analytic Solution')

ax1.legend()
ax2.legend()
plt.tight_layout()

# ---------------- SCIPY SOLVER VS ANALYTIC SOLUTION ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,8))

ax1.plot(t_arr, x1_as, label = f'Mass 1 Analytic Solution', color = 'C2')
ax2.plot(t_arr, x2_as, label = f'Mass 2 Analytic Solution', color = 'C2')

#---------- SCIPY solve_ivp SOLUTIONS ------------
ax1.plot(t_arr, x1sp, label = f'Mass 1 solve_ivp RK45', color = 'C3')
ax2.plot(t_arr, x2sp, label = f'Mass 2solve_ivp RK45', color = 'C3')

ax1.legend()
ax2.legend()
plt.tight_layout()

# ---------------- SCIPY SOLVER VS MY RK4 CODE ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,8))

ax1.plot(t_arr, x1r, label = f'Mass 1 RK4', color = 'C1')
ax2.plot(t_arr, x2r, label = f'Mass 2 RK4', color = 'C1')

#---------- SCIPY solve_ivp SOLUTIONS ------------
ax1.plot(t_arr, x1sp, label = f'Mass 1 solve_ivp RK45', color = 'C3')
ax2.plot(t_arr, x2sp, label = f'Mass 2solve_ivp RK45', color = 'C3')

ax1.legend()
ax2.legend()
plt.tight_layout()

plt.show()

