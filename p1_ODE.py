import matplotlib.pyplot as plt
import numpy as np

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
m2 = 2
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

# ----------------------- SETUP FOR CALCULATIONS -----------------------
tmax = 20
t_arr = np.arange(start = 0, stop = tmax + dt, step = dt)
x1 = np.array([x10])
x2 = np.array([x20])
v1 = np.array([v10])
v2 = np.array([v20])


# ----------------------- ARRAY CALCULATIONS FOR FIGURES -----------------------
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
    x1 = np.append(x1, x1np1)
    x2 = np.append(x2, x2np1)
    v1 = np.append(v1, v1np1)
    v2 = np.append(v2, v2np1)



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


plt.figure()
fig, ax = plt.subplots()
ax.plot(t_arr, x1, label = f'Mass 1 Euler')
ax.plot(t_arr, x2, label = f'Mass 2 Euler')
plt.legend()

#plt.tight_layout()
#plt.show()

#plt.figure()
#fig, ax = plt.subplots()
ax.plot(t_arr, x1r, label = f'Mass 1 RK4')
ax.plot(t_arr, x2r, label = f'Mass 2 RK4')
plt.legend()

plt.tight_layout()
plt.show()

# TO DO:
#   Energy Conservation
#   Test against analytic solutions
#   Compare with PEP8 formatting guidelines
#   Make into separate files (Obviously)