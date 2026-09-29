import matplotlib.pyplot as plt
import numpy as np

import methods as meth

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
    into recursive Reimann integrator.

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
    exp_val_final = meth.recReimann(f, xmin, xmax, dx, x_func = x_func, psi = psi, **kwargs)
    return exp_val_final

def expecFunc(psi, xmin, xmax, dx, x_func = None, **kwargs):
    """
    Function that calculates the expectation value of a given quantity (default is x) at a single 
    position to be plugged into Reimann integrator.

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
    exp_val_final = meth.Reimann(f, xmin, xmax, dx, x_func = x_func, psi = psi, **kwargs)
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
psi_x = psi_isw(x_arr, a, 1)
psi_x_num = np.array([])
A_tot = 0
for i in range(len(x_arr)):
    psi_x_num = np.append(psi_x_num, A_tot)
    A_tot += meth.Reimann(dpsi_isw, x_arr[i], step, a = a, n = 1)

plt.figure()
fig, ax = plt.subplots()
ax.plot(x_arr, psi_x, label = r'Analytic $\psi(x)$')
ax.plot(x_arr, psi_x_num, label = r'Reimann Integrated $\psi(x)$')
ax.set_xlabel('Energy Level')
ax.set_ylabel(r'$\langle x\rangle$')
plt.legend()
plt.tight_layout()
plt.show()

# Energy levels to find <x^2> for
# n_lst = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# # List of expectation values
# exp_lst = np.array([])
# for i in n_lst:
#     # get the expectation value of x squared
#     exp_x_squared = meth.expecVal(psi_isw, 0, a, step, x_func = squared, n = i, a = a)
#     # add to list
#     exp_lst = np.append(exp_lst, exp_x_squared)

# plt.figure()
# fig, ax = plt.subplots()
# ax.scatter(n_lst, exp_lst)
# ax.set_xlabel('Energy Level')
# ax.set_ylabel(r'$\langle x\rangle$')
# plt.tight_layout()
# plt.show()

# Griffiths 2.45
# Nodes at n = 3: a/3, 2a/3
x1 = a/3
x2 = (2 * a)/3

dpsi_32 = dpsi_isw(x = x2, a = a, n = 3)
dpsi_31 = dpsi_isw(x = x1, a = a, n = 3)

psi_21 = psi_isw(x = x1, a = a, n = 2)
psi_22 = psi_isw(x = x2, a = a, n = 2)
m = 1
E3 = E_isw(3, a)
E2 = E_isw(2, a)

def psitpsi(xmin, n1, n2, psi, a, **kwargs):
    psi1 = psi_isw(xmin, a, n1)
    psi2 = psi_isw(xmin, a, n2)
    psi_prod = psi1 * psi2
    return psi_prod

int_psiprod = meth.reimannRec(psitpsi, x1, x2, step, n1 = 3, n2 = 2, a = a, psi = psi_isw)
lhs = (dpsi_32 * psi_22) - (dpsi_31 * psi_21)
rhs = ((2*m)/hbar) * (E2 - E3) * int_psiprod



#print(E2)
#print(E3)
# print(f'Left: {lhs}')
# print(f'Right: {rhs}')

# To Do:
#   Improve comments– got kinda lazy there
#   Look into Griffiths problem further– do more psi states probably
#   Try part a with the Euler/RK4 calculators and see how the results match up with the integrator
