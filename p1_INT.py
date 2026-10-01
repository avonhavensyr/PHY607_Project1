import matplotlib.pyplot as plt
import numpy as np
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

Numerical Integrator: Infinite Square Well
"""

# –––––––––––– PROBLEM-SPECIFIC CONDITIONS AND FUNCTIONS ––––––––––––

# CONSTANTS
# reduced planck constant in natural units
hbar = 1
# well width
a = 5
# change in x
step = 0.1
# particle mass in eV
m = 1

def expecVal(psi, xmin, xmax, dx, x_func = None, **kwargs):
    """
    Function that calculates the expectation value of a given quantity (default is x) to be plugged 
    into recursive Riemann integrator.

    Keyword Arguments
    psi (function): wavefunction
    xmin (int/float): initial position wavefunction is evaluated at
    xmax (int/float): final position wavefunction is evaluated at
    dx (int/float): change in x between steps
    x_func (function): function that you are taking the expectation value of-- x if not specified
    """
    # define function to integrate using the standard formula for expectation value
    def f(xmin, psi, x_func, **kwargs):
        # get the value of the wavefunction at xmin
        wf = psi(xmin, **kwargs)
        # get the complex conjugate of the wavefunction at xmin
        wfi = np.conjugate(wf)
        # take the expectation value of specifically entered f(x) (ie: x_func)
        if x_func is not None:
            exp_val = wfi * x_func(xmin) * wf
        # default case: take the expectation value of x
        else: 
            exp_val = wfi * xmin * wf
        return exp_val
    exp_val_final = meth.recRiemann(f, xmin, xmax, dx, x_func = x_func, psi = psi, **kwargs)
    return exp_val_final

def expecFunc(psi, xmin, xmax, dx, x_func = None, **kwargs):
    """
    Function that calculates the expectation value of a given quantity (default is x) at a single 
    position to be plugged into Riemann integrator.

    Keyword Arguments
    psi (function): wavefunction
    xmin (int/float): initial position wavefunction is evaluated at
    xmax (int/float): final position wavefunction is evaluated at
    dx (int/float): change in x between steps
    x_func (function): function that you are taking the expectation value of-- x if not specified
    """
    # define function to integrate using the standard formula for expectation value
    def f(xmin, psi, x_func, **kwargs):
        # get the value of the wavefunction at xmin
        wf = psi(xmin, **kwargs)
        # get the complex conjugate of the wavefunction at xmin
        wfi = np.conjugate(wf)
        # take the expectation value of specifically entered f(x) (ie: x_func)
        if x_func is not None:
            exp_val = wfi * x_func(xmin) * wf
        # default case: take the expectation value of x
        else: 
            exp_val = wfi * xmin * wf
        return exp_val
    exp_val_final = meth.Riemann(f, xmin, xmax, dx, x_func = x_func, psi = psi, **kwargs)
    return exp_val_final

def psi_isw(x, a, n, **kwargs):
    """
    Position wavefunction for the infinite square well

    Keyword Arguments:
    x (int/float): position wavefunction is evaluated at
    a (int/float): width of the well
    n (int): energy level 
    """
    psi = np.sqrt(2 / a) * np.sin((n * np.pi * x) / a)
    return psi

def dpsi_isw(x, a, n, **kwargs):
    """
    Derivative of the position wavefunction for the infinite square well

    Keyword Arguments:
    x (int/float): position wavefunction is evaluated at
    a (int/float): width of the well
    n (int): energy level     
    """
    dpsi = np.sqrt(2 / a) * ((n * np.pi) / a) * np.cos((n * np.pi * x) / a)
    return dpsi

def d2psi_is2(x, a, n, **kwargs):
    """
    Second derivative of the position wavefunction for the infinite square well

    Keyword Arguments:
    x (int/float): position wavefunction is evaluated at
    a (int/float): width of the well
    n (int): energy level     
    """
    d2psi = np.sqrt(2 / a) * (((n * np.pi) / a)**2) * -1 * np.sin((n * np.pi * x) / a)
    return d2psi

def E_isw(n, a, m = 1, **kwargs):
    """
    Energy for the infinite square well

    Keyword Arguments:
    n (int): energy level     
    a (int/float): width of the well
    m (int/float): mass of the particle-- default 1
    """
    return (((n * np.pi * hbar)/a) ** 2) * (1 / (2 * m))

def squared(x):
    """
    x squared

    Keyword Arguments:
    x (int/float): value you want to square
    """
    return x**2

# Trivial Test
step = 0.01
x_arr = np.arange(start = -a, stop = a + step, step = step)
psi_x1 = psi_isw(x_arr, a, 1)
psi_x1_num = np.array([])

A_tot = 0
for i in range(len(x_arr)):
    psi_x1_num = np.append(psi_x1_num, A_tot)
    A_tot += meth.Riemann(dpsi_isw, x_arr[i], step, a = a, n = 1)


# ----------------------------- ERRORS ---------------------------
# Global truncation error for the ground state
gerr1 = np.abs(psi_x1 - psi_x1_num)
# Local truncation error for the ground state
lerr1 = np.array([])
# copied from ODE side
dx_arr = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

# make arrays for the first three energy levels
avg_lerr1 = np.array([])
avg_lerr2 = np.array([])
avg_lerr3 = np.array([])
avg_lerr4 = np.array([])

level_lst = [avg_lerr1, avg_lerr2, avg_lerr3]
for lvl, arr in enumerate(level_lst):
    for dx in dx_arr:
        lerr = np.array([0])
        for i in range(len(x_arr) - 1):
            # numerical solution
            psinp1 = meth.Riemann(dpsi_isw, x_arr[i], dx, a = a, n = lvl + 1)
            # change in psi_isw when xf-xi = dx
            psi_diff = psi_isw(x_arr[i] + dx, a, 1) - psi_isw(x_arr[i], a, 1)
            # local error
            lerr = np.append(lerr, np.abs(psinp1 - psi_diff))
        level_lst[lvl] = np.append(level_lst[lvl], np.mean(lerr))

# Plot quadratic curves that align with the expected local error for a reimann integrator
c1 = 0.08
th_lerr1 = c1 * (dx_arr ** 2)



fig, ax = plt.subplots(figsize = (8, 5))
ax.set_title(r'Local Error of the Ground State')
ax.scatter(dx_arr, level_lst[0], label = rf'$n=1$')
ax.plot(dx_arr, th_lerr1, linestyle = '--', label = rf'{c1}$\Delta x^2$')

ax.set_xlabel(r'$\Delta x$')
ax.set_ylabel(r'Local Error')

ax.legend()

plt.tight_layout()
plt.savefig('n_lerr_comp.png')


fig, ax = plt.subplots(figsize = (8, 5))
ax.set_title(r'Log Local Error for $n=1,\ 2,\ 3$')
ax.loglog(dx_arr, level_lst[0], label = rf'$n=1$')
ax.loglog(dx_arr, level_lst[1], label = rf'$n=2$')
ax.loglog(dx_arr, level_lst[2], label = rf'$n=3$')
ax.loglog(dx_arr, level_lst[3], label = rf'$n=4$')


ax.set_xlabel(r'$\Delta x$')
ax.set_ylabel(r'Local Error')

ax.legend()

plt.tight_layout()
plt.savefig('n_lerr_comp.png')
# ------------------- TESTED PROPERTIES/LIMITING CASES -------------------
# ---------------- II: Griffiths 2.4-- Convergence of <x^2> for high n-------------------

# Energy levels to find <x^2> for
n_lst = np.arange(1, 26)       # go up to n=20
step = 0.02
# List of expectation values
exp_lst = np.array([])
for i in n_lst:
    # get the expectation value of x squared
    exp_x_squared = expecVal(psi_isw, 0, a, step, x_func = squared, n = i, a = a)
    # add to list
    exp_lst = np.append(exp_lst, exp_x_squared)
# value the expectation value should converge to
con_val = (a**2) / 3
# make figure
fig, ax = plt.subplots()
ax.axhline(y = con_val, linestyle = '--', label = r'$\langle x^2\rangle=\frac{a^2}{3}$', color = 'C1', zorder=1)
ax.scatter(n_lst, exp_lst, label = r'Numerical $\langle x^2\rangle_n$ Values', marker = 'o')
ax.set_xlabel('n')
ax.set_ylabel(r'$\langle x\rangle$')
ax.legend()
plt.tight_layout()
plt.savefig('xsq_conv.png')

plt.show()

# ---------------- II: Griffiths 2.45 -- Property of Nodes -------------------
# Nodes at n = 3: a/3, 2a/3
# x1 = a/3
# x2 = (2 * a)/3
# # Tuples:
# pairs = [(3, 2), (5, 6), (8, 9)]
# # Node Pairs:
# nodes = [(a/3, (2*a)/3), ((2 * a)/5, (4 * a)/5), ((4 * a)/9, (8 * a)/9)]

# for (i, j) in pairs:
#     dpsi_32 = dpsi_isw(x = x2, a = a, n = 3)
#     psi_31 = dpsi_isw(x = x1, a = a, n = 3)

# psi_21 = psi_isw(x = x1, a = a, n = 2)
# psi_22 = psi_isw(x = x2, a = a, n = 2)
# m = 1
# E3 = E_isw(3, a)
# E2 = E_isw(2, a)

# def psitpsi(xmin, n1, n2, psi, a, **kwargs):
#     psi1 = psi_isw(xmin, a, n1)
#     psi2 = psi_isw(xmin, a, n2)
#     psi_prod = psi1 * psi2
#     return psi_prod

# int_psiprod = meth.recRiemann(psitpsi, x1, x2, step, n1 = 3, n2 = 2, a = a, psi = psi_isw)
# lhs = (dpsi_32 * psi_22) - (dpsi_31 * psi_21)
# rhs = ((2*m)/hbar) * (E2 - E3) * int_psiprod

# fig, ax = plt.subplots()
# ax.set_title(rf'Riemann Integrator vs Analytic Solution: $\Delta x= $ {step}')
# ax.plot(x_arr, psi_x1, label = r'Analytic $\psi(x)$')
# ax.plot(x_arr, psi_x1_num, label = r'Riemann Integrated $\psi(x)$')
# ax.set_xlabel('x')
# ax.set_ylabel(r'$\langle x\rangle$')
# plt.legend()
# plt.tight_layout()
# plt.savefig('psiint_v_psian.png')
# plt.show()

# print(E2)
# print(E3)
# print(f'Left: {lhs}')
# print(f'Right: {rhs}')
