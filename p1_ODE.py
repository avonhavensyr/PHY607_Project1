import matplotlib.pyplot as plt
import numpy as np

"""
PROJECT I
Draft: Includes everything for the ODE problem in one file. 
Separation between module and main file will be implemented later.
"""

"""
ODE Solver: Coupled Harmonic Oscillator
"""
# Notes
# ma1 = -kx1 -k(x1-x2)
# ma2 = -2kx2 + kx1

# U1 = 1/2(kx1^2) + 1/2(k(x1-x2)^2)
#    = k/2[(x1^2) + (x1^2 - 2x1x2 + x2^2)]
#    = kx2^2 - kx1x2 + k/2x2^2 
# U2 = 1/2k(2x2^2) +1/2k(x1^2)
# going to take on pen and paper

# Note: basically re-writing my script because there were issues with the
# recursion depth that caused issues with my time steps, and I don't feel 
# like dealing with that

# CONSTANTS
m1 = 1
m2 = 2
k = 1

# INITIAL CONDITIONS
x10 = 1         # mass 1: initial displacement
x20 = 1         # mass 2: initial displacement
v10 = 1         # mass 1: initial velocity
v20 = 1         # mass 2: initial velocity

t0 = 0
dt = 0.2

dv1 = "(1/m) * ((-2 * k * x1) + (k * x2))"
dv2 = "(1/m) * ((-2 * k * x2) + (k * x1))"

def f(xi, xj, mi):
    # function that returns the acceleration
    dvi = ((1 / mi) * ((-2 * k * xi) + (k * xj)))
    return dvi

def Euler(m, v, x, xi, dt):
    """
    Function that evaluates a new value of a function and its derivative
    based previous input values and returns the new state as a vector using
    the symplectic Euler method
    """
    # going to use symplectic method
    # starting with the numbers as singular values rather than as arrays like
    # previous Euler solvers
    dv = f(x, xi, m)
    v_new = v + (dv * dt)       # velocity from Euler method
    x_new = x + (v_new * dt)    # position from Euler method
    return np.array([x_new, v_new])

def RK4(mi, mj, vi, vj, xi, xj, dt):
    """
    Function that evaluates a new value of a function and its derivative
    based previous input values and returns the new state as a vector using
    the fourth-order Runge-Kutta method
    """
    k1xi = vi
    k1vi = dt * f(xi, xj, mi)
    k1xj = vj
    k1vj = dt * f(xj, xi, mj)

    k2xi = vi + ((dt/2) * k1vi)
    k2vi = f((xi + ((dt/2) * k1xi)), (xj + ((dt/2) * k1xj)), mi)
    k2xj = vj + ((dt/2) * k1vj)
    k2vj = f((xj + ((dt/2) * k1xj)), (xi + ((dt/2) * k1xi)), mj)

    k3xi = vi + ((dt/2) * k2vi)
    k3vi = f((xi + ((dt/2) * k2xi)),(xj + ((dt/2) * k2xj)), mi)
    k3xj = vj + ((dt/2) * k2vj)
    k3vj = f((xj + ((dt/2) * k2xj)),(xi + ((dt/2) * k2xi)), mj)

    k4xi = vi + ((dt/2) * k3vi)
    k4vi = f((xi + ((dt/2) * k3xi)),(xj + ((dt/2) * k3xj)), mi)
    k4xj = vj + ((dt/2) * k3vj)
    k4vj = f((xj + ((dt/2) * k3xj)),(xi + ((dt/2) * k3xi)), mj)

    xi_new = xi + ((dt/6) * (k1xi + (2*k2xi) + (2*k3xi) + k4xi))
    vi_new = vi + ((dt/6) * (k1vi + (2*k2vi) + (2*k3vi) + k4vi))

    xj_new = xj + ((dt/6) * (k1xj + (2*k2xj) + (2*k3xj) + k4xj))
    vj_new = vj + ((dt/6) * (k1vj + (2*k2vj) + (2*k3vj) + k4vj))

    return (np.array([xi_new, vi_new]), np.array([xj_new, vj_new]))

tmax = 20
t_arr = np.arange(start = 0, stop = tmax + dt, step = dt)
x1 = np.array([x10])
x2 = np.array([x20])
v1 = np.array([v10])
v2 = np.array([v20])

for i in range(len(t_arr)-1):
    x1n = x1[-1]                                # most recent (nth) position for m1
    x2n = x2[-1]                                # most recent (nth) position for m2

    v1n = v1[-1]                                # most recent (nth) velocity for m1
    v2n = v2[-1]                                # most recent (nth) velocity for m2

    s1np1 = Euler(m1, v1n, x1n, x2n, dt)        # state vector for the n+1th time
    x1np1 = s1np1[0]                            # n+1th position for m1
    v1np1 = s1np1[1]                            # n+1th velocity for m1
    x1 = np.append(x1, x1np1)                   # add new values to arrays
    v1 = np.append(v1, v1np1)                   # ^^^

    s2np1 = Euler(m2, v2n, x2n, x1n, dt)        # state vector for the n+1th time
    x2np1 = s2np1[0]                            # n+1th position for m2
    v2np1 = s2np1[1]                            # n+1th velocity for m2
    x2 = np.append(x2, x2np1)                   # add new values to arrays
    v2 = np.append(v2, v2np1)                   # ^^^


x1r = np.array([x10])
x2r = np.array([x20])
v1r = np.array([v10])
v2r = np.array([v20])

for i in range(len(t_arr)-1):
    x1n = x1r[-1]                                # most recent (nth) position for m1
    x2n = x2r[-1]                                # most recent (nth) position for m2

    v1n = v1r[-1]                                # most recent (nth) velocity for m1
    v2n = v2r[-1]                                # most recent (nth) velocity for m2

    s1np1, s2np1 = RK4(m1, m2, v1n, v2n, x1n, x2n, dt)        # state vector for the n+1th time
    x1np1 = s1np1[0]                            # n+1th position for m1
    v1np1 = s1np1[1]                            # n+1th velocity for m1
    x1r = np.append(x1r, x1np1)                   # add new values to arrays
    v1r = np.append(v1r, v1np1)                   # ^^^
    x2np1 = s2np1[0]                            # n+1th position for m2
    v2np1 = s2np1[1]                            # n+1th velocity for m2
    x2r = np.append(x2r, x2np1)                   # add new values to arrays
    v2r = np.append(v2r, v2np1)                   # ^^^

#print(len(t_arr))
#print(len(x1))

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