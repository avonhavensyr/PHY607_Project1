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
# discrete x values
dx_arr = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
x_arr = [np.arange(start =-a, stop = a+dx, step = dx) for dx in dx_arr]
sidx = 0
lidx = -2
dx_s = dx_arr[sidx]
dx_l = dx_arr[lidx]
# make dictionaries to hold the analytical and numerical values for each n and dx
# Find errors
# plot errors
# Solve limiting cases
psi_an_dict = {dx: [] for dx in dx_arr}
psi_num_dict = {dx: [] for dx in dx_arr}
# get the numerical and analytical values for each psi_n
for i, dx in enumerate(dx_arr):
    for n in range(n_levels):
        psi_an_dict[dx].append(np.array([]))
        psi_num_dict[dx].append(np.array([]))
        A_tot = 0
        for x, arr in enumerate(x_arr[i]):
            A_tot = meth.Riemann(dpsi_isw, arr, dx, A=A_tot, a = a, n = n + 1)
            psi_an_dict[dx][n] = np.append(psi_an_dict[dx][n], psi_isw(arr, a, n+1))
            psi_num_dict[dx][n] = np.append(psi_num_dict[dx][n], A_tot)






# ----------------------------- ERRORS ---------------------------
# Global truncation error for the ground state
gerr_dict = {dx: [] for dx in dx_arr}
avg_gerr_dict = {dx: [] for dx in dx_arr}

for i, dx in enumerate(dx_arr):
    for n in range(n_levels):
        gerr_dict[dx].append(np.abs(psi_an_dict[dx][n] - psi_num_dict[dx][n]))
        avg_gerr_dict[dx].append(np.mean(np.abs(psi_an_dict[dx][n] - psi_num_dict[dx][n])))


lerr_dict = {dx: [] for dx in dx_arr}
avg_lerr_dict = {dx: [] for dx in dx_arr}

for i, dx in enumerate(dx_arr):
    for n in range(5):
        lerr = []
        for x in range(len(x_arr[i]) - 1):
            psi_diff = psi_an_dict[dx][n][x + 1] - psi_an_dict[dx][n][x]
            psi_num = meth.Riemann(dpsi_isw, x_arr[i][x], dx, a=a, n=n + 1)
            lerr.append(np.abs(psi_num - psi_diff))
        lerr_dict[dx].append(np.array(lerr))
        avg_lerr_dict[dx].append(np.mean(lerr))

lerr_lst = [[lerr_dict[dx][n] for dx in dx_arr] for n in range(5)]
gerr_lst = [[gerr_dict[dx][n] for dx in dx_arr] for n in range(5)]

avg_lerr_lst = [[avg_lerr_dict[dx][n] for dx in dx_arr] for n in range(5)]
avg_gerr_lst = [[avg_gerr_dict[dx][n] for dx in dx_arr] for n in range(5)]

dx_short = dx_arr[:5]

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 8))

O_11 = dx_short
O_21 = dx_short ** 2


ax1.loglog(dx_short, 3.5 * O_21, label = r'$\mathcal{O}(\Delta t)$',color = 'black',ls='--')
ax2.loglog(dx_short, 1.5 * O_11, label = r'$\mathcal{O}(\Delta t^2)$',color = 'black',ls='--')

ax1.set_title('Average Local Error')
ax2.set_title('Average Global Error')
ax3.set_title('Local Error Over x')
ax4.set_title('Global Error Over x')

for i in range(5):
    ax1.loglog(dx_short, avg_lerr_lst[i][:5], label=rf'$n={i+1}$')
    ax2.loglog(dx_short, avg_gerr_lst[i][:5], label=rf'$n={i+1}$')
    ax3.loglog(x_arr[sidx][:-1], lerr_dict[dx_s][i], color=f'C{i}', linestyle='-',  label = rf'$n=${i+1}: $\Delta x=${dx_s}')
    ax3.loglog(x_arr[lidx][:-1], lerr_dict[dx_l][i], color=f'C{i}', linestyle='--', label = rf'$n=${i+1} $\Delta x=${dx_l}')
    ax4.loglog(x_arr[sidx], gerr_dict[dx_s][i], color=f'C{i}', linestyle='-',  label = rf'$n=${i+1}: $\Delta x=${dx_s}')
    ax4.loglog(x_arr[lidx], gerr_dict[dx_l][i], color=f'C{i}', linestyle='--', label = rf'$n=${i+1} $\Delta x=${dx_l}')

ax1.set_xlabel(r'$\Delta x$')
ax2.set_xlabel(r'$\Delta x$')
ax3.set_xlabel(r'$x$')
ax4.set_xlabel(r'$x$')
ax1.legend(loc = 'lower left', fontsize = 9)
ax2.legend(loc = 'lower left', fontsize = 9)
ax3.legend(loc = 'lower left', fontsize = 9)
ax4.legend(loc = 'lower left', fontsize = 9)
plt.tight_layout()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 8), sharex = True, sharey = True)


fig.supxlabel('x')
fig.supylabel(r'$\psi_n(x)$')


ax1.set_title(rf'$n=1$')
ax1.plot(x_arr[sidx],psi_an_dict[dx_s][0], label = rf'Analytical: $\Delta x =${dx_s}')
ax1.plot(x_arr[sidx],psi_num_dict[dx_s][0], label = rf'Numerical: $\Delta x =${dx_s}', ls = '--')
ax1.plot(x_arr[lidx],psi_an_dict[dx_l][0], label = rf'Analytical: $\Delta x =${dx_l}', ls = '--')
ax1.plot(x_arr[lidx],psi_num_dict[dx_l][0], label = rf'Numerical: $\Delta x =${dx_l}', ls = '--')

ax2.set_title(rf'$n=2$')
ax2.plot(x_arr[sidx],psi_an_dict[dx_s][1], label = rf'Analytical: $\Delta x =${dx_s}')
ax2.plot(x_arr[sidx],psi_num_dict[dx_s][1], label = rf'Numerical: $\Delta x =${dx_s}', ls = '--')
ax2.plot(x_arr[lidx],psi_an_dict[dx_l][1], label = rf'Analytical: $\Delta x =${dx_l}', ls = '--')
ax2.plot(x_arr[lidx],psi_num_dict[dx_l][1], label = rf'Numerical: $\Delta x =${dx_l}', ls = '--')

ax3.set_title(rf'$n=3$')
ax3.plot(x_arr[sidx],psi_an_dict[dx_s][2], label = rf'Analytical: $\Delta x =${dx_s}')
ax3.plot(x_arr[sidx],psi_num_dict[dx_s][2], label = rf'Numerical: $\Delta x =${dx_s}', ls = '--')
ax3.plot(x_arr[lidx],psi_an_dict[dx_l][2], label = rf'Analytical: $\Delta x =${dx_l}', ls = '--')
ax3.plot(x_arr[lidx],psi_num_dict[dx_l][2], label = rf'Numerical: $\Delta x =${dx_l}', ls = '--')

ax4.set_title(rf'$n=4$')
ax4.plot(x_arr[sidx],psi_an_dict[dx_s][3], label = rf'Analytical: $\Delta x =${dx_s}')
ax4.plot(x_arr[sidx],psi_num_dict[dx_s][3], label = rf'Numerical: $\Delta x =${dx_s}', ls = '--')
ax4.plot(x_arr[lidx],psi_an_dict[dx_l][3], label = rf'Analytical: $\Delta x =${dx_l}', ls = '--')
ax4.plot(x_arr[lidx],psi_num_dict[dx_l][3], label = rf'Numerical: $\Delta x =${dx_l}', ls = '--')

ax1.legend(loc = 'lower left', fontsize = 9)
ax2.legend(loc = 'lower left', fontsize = 9)
ax3.legend(loc = 'lower left', fontsize = 9)
ax4.legend(loc = 'lower left', fontsize = 9)

plt.tight_layout()

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

#make figure
fig, ax = plt.subplots()
ax.axhline(y = con_val, linestyle = '--', label = r'$\langle x^2\rangle=\frac{a^2}{3}$', color = 'C1', zorder=1)
ax.scatter(n_lst, exp_lst, label = r'Numerical $\langle x^2\rangle_n$ Values', marker = 'o')
ax.set_xlabel('n')
ax.set_ylabel(r'$\langle x\rangle$')
ax.legend()
plt.tight_layout()


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



plt.show()
