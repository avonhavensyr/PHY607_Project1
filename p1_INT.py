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

def psiProd(xmin, n1, n2, psi, a, **kwargs):
    psi1 = psi_isw(xmin, a, n1)
    psi2 = psi_isw(xmin, a, n2)
    psi_prod = psi1 * psi2
    return psi_prod

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

n_levels = 10       # number of energy levels
# analytical states
n_states = [np.array([]) for i in range(n_levels)]
# numerical approximations
n_approx = [np.array([]) for i in range(n_levels)]
# discrete x values
x_arr = np.arange(start = -a, stop = a + step, step = step)

# get the numerical and analytical values for each psi_n
for lvl, arr in enumerate(n_states):
    psi_an = psi_isw(x_arr, a, lvl + 1)
    n_states[lvl] = psi_an
    psi_num = np.array([])
    A_tot = 0
    for i in range(len(x_arr)):
        psi_num = np.append(psi_num, A_tot)
        A_tot += meth.Riemann(dpsi_isw, x_arr[i], step, a = a, n = lvl + 1)
    n_approx[lvl] = psi_num

#ax.plot(x_arr, n_states[0], label = 'Ground State Analytical')
#ax.plot(x_arr, n_approx[0], label = 'Ground State Numerical')



# ----------------------------- ERRORS ---------------------------
# Global truncation error for the ground state
n_gerr = [np.array([]) for i in range(n_levels)]
for lvl, arr in enumerate(n_states):
    gerr = np.abs(n_states[lvl] - n_approx[lvl])
    n_gerr = np.append(n_gerr, gerr)

print(n_gerr)
# Local truncation error for the ground state
lerr1 = np.array([])
# copied from ODE side
dx_arr = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

n_lerr = [np.array([]) for i in range(n_levels)]
for lvl, arr in enumerate(n_lerr[1:]):
    for dx in dx_arr:
        lerr = np.array([0])
        for i in range(len(x_arr) - 1):
            # numerical solution
            psinp1 = meth.Riemann(dpsi_isw, x_arr[i], dx, a = a, n = lvl + 1)
            # change in psi_isw when xf-xi = dx
            psi_diff = psi_isw(x_arr[i] + dx, a, 1) - psi_isw(x_arr[i], a, 1)
            # local error
            lerr = np.append(lerr, np.abs(psinp1 - psi_diff))
        n_lerr[lvl] = np.append(n_lerr[lvl], np.mean(lerr))

# Plot quadratic curves that align with the expected local error for a reimann integrator
c1 = 0.08
th_lerr1 = c1 * (dx_arr ** 2)

fig, ax = plt.subplots(figsize = (8, 5))
ax.set_title(r'Local Error of the Ground State')
ax.scatter(dx_arr, n_states[0], label = rf'$n=1$')
ax.plot(dx_arr, th_lerr1, linestyle = '--', label = rf'{c1}$\Delta x^2$')

ax.set_xlabel(r'$\Delta x$')
ax.set_ylabel(r'Local Error')

ax.legend()

plt.tight_layout()
plt.savefig('n_lerr_comp.png')

ex_n_states = n_states[1:]      # extract only the exited states
fig, ax = plt.subplots(figsize = (8, 5))
ax.set_title(r'Log Local Error for Excited States')
for lvl, arr in enumerate(ex_n_states):
    ax.loglog(dx_arr, ex_n_states[lvl], label = rf'$n=${lvl+2}')
ax.set_xlabel(r'$\Delta x$')
ax.set_ylabel(r'Local Error')

ax.legend()

plt.tight_layout()
plt.savefig('NEWn_lerr_comp.png')


# ------------------- TESTED PROPERTIES/LIMITING CASES -------------------
# ---------------- II: Griffiths 2.4-- Convergence of <x^2> for high n-------------------



# make figure
# fig, ax = plt.subplots()
# ax.axhline(y = con_val, linestyle = '--', label = r'$\langle x^2\rangle=\frac{a^2}{3}$', color = 'C1', zorder=1)
# ax.scatter(n_lst, exp_lst, label = r'Numerical $\langle x^2\rangle_n$ Values', marker = 'o')
# ax.set_xlabel('n')
# ax.set_ylabel(r'$\langle x\rangle$')
# ax.legend()
# plt.tight_layout()
# plt.savefig('xsq_conv.png')

# plt.show()

# ---------------- II: Griffiths 2.45 -- Property of Nodes -------------------
# Nodes at n = 3: a/3, 2a/3
# Tuples:
pairs = [(3, 2), (5, 6), (8, 9)]
# Node Pairs:
nodes = [(a/3, (2*a)/3), ((2 * a)/5, (4 * a)/5), ((4 * a)/9, (8 * a)/9)]

dpsi1 = []
psi2 = []

lhs_lst = []
rhs_lst = []

for i, ((n1, n2), (x1, x2)) in enumerate(zip(pairs,nodes)):
    # dpsi1 values
    dp11 = dpsi_isw(x = x2, a = a, n = n1)
    dp12 = dpsi_isw(x = x1, a = a, n = n1)
    dpsi1.append((dp11, dp12))
    # psi1 values
    p21 = psi_isw(x = x2, a = a, n = n2)
    p22 = psi_isw(x = x1, a = a, n = n2)
    psi2.append((p21, p22))
    # energies for n1 and n2
    En1 = E_isw(n1, a)
    En2 = E_isw(n2, a)
    int_psiprod = meth.recRiemann(psiProd, x1, x2, step, n1 = 3, n2 = 2, a = a, psi = psi_isw)
    lhs = (dp12 * p22) - (dp11 * p21)
    rhs = ((2*m)/hbar) * (En1 - En2) * int_psiprod

    lhs_lst.append(lhs)
    rhs_lst.append(rhs)

print(f'Pairs: {pairs}')
print(f'Nodes: {nodes}')
print(f'Left Hand Side: {lhs_lst}')
print(f'Right Hand Side: {rhs_lst}')
for p, pair in enumerate(pairs):
    print(f'Difference for pair{p+1}: {lhs_lst[p] - rhs_lst[p]}')



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

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 8))

ax1.set_title(rf'$n=1$')
ax1.plot(x_arr, n_states[0], label = rf'Analytical', ls = '-')
ax1.plot(x_arr, n_approx[0], label = rf'Numerical', ls = '--')
ax1.legend()

ax2.set_title(rf'$n=1$')
ax2.plot(x_arr, n_states[1], label = rf'Analytical', ls = '-')
ax2.plot(x_arr, n_approx[1], label = rf'Numerical', ls = '--')
ax2.legend()

ax3.set_title(rf'$n=1$')
ax3.plot(x_arr, n_states[2], label = rf'Analytical', ls = '-')
ax3.plot(x_arr, n_approx[2], label = rf'Numerical', ls = '--')
ax3.legend()

ax4.set_title(rf'$n=1$')
ax4.plot(x_arr, n_states[3], label = rf'Analytical', ls = '-')
ax4.plot(x_arr, n_approx[3], label = rf'Numerical', ls = '--')
ax4.legend()

plt.tight_layout()
plt.savefig('NEWint_comparisons.png')
plt.show()