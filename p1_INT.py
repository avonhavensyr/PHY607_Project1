import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const


"""
PROJECT I
"""

"""
Numerical Integrator: Infinite Square Well
"""
# Switching from harmonic oscillator to make the integrator finite

# Note for instructor: I just got a Macbook from the school and discovered
# the "option" key– all em-dash characters present in code comments are
# manually typed by me hitting the option key with ZERO assistance from AI

# going to start with a simple Reimann sum– I hate hate hate doing
# the easiest option provided, but I need to make sure I can submit
# complete by Tuesday
# Reduced Planck Constant, SI units

# –––––––––––– INTEGRATOR ––––––––––––

def Reimann(f, xmin, xmax, dx, A=0, **kwargs):
    """
    f (function): Function being integrated
    xmin (int/float): Value being integrated over
    xmax (int/float): Upper integration bound
    dx (int/floar): Step-size
    A (int): starting area under curve
    """
    h = f(xmin, **kwargs)           # Find the height
    A += h * dx                     # Calculate the area of the Reimann sum rectangle
    if xmin > xmax:
        return A
    else:
        return Reimann(f, xmin + dx, xmax, dx, A, **kwargs)

# –––––––––––– PROBLEM-SPECIFIC CONDITIONS AND FUNCTIONS ––––––––––––

hbar = 1           # Reduced Planck Constant in natural units

def psi_isw(x, a, n, **kwargs):
    """
    Position wavefunction for the infinite square well
    x (int/float): position wavefunction is evaluated at
    a (int/float): width of the well
    n (int): energy level 
    """
    psi = np.sqrt(2 / a) * np.sin((n * np.pi * x) / a)
    return psi

def dpsi_isw(x, a, n, **kwargs):
    dpsi = np.sqrt(2 / a) * ((n * np.pi) / a) * np.cos((n * np.pi * x) / a)
    return dpsi

def d2psi_is2(x, a, n, **kwargs):
    d2psi = np.sqrt(2 / a) * (((n * np.pi) / a)**2) * -1 * np.sin((n * np.pi * x) / a)

def E_isw(n, a, m = 1, **kwargs):
    return (((n * np.pi * hbar)/a) ** 2) * (1 / (2 * m))

def psit_isw(x, a, n, **kwargs):
    c = (complex(0, -1) * hbar)
    # psi_x = psi_isw(x) * 

def squared(x):
    return x**2     #x^2

# finding it impossible to make generalized expectation value function, so I'm just going to make four of them
def expecVal(x, psi, x_func = None, **kwargs):
    """
    Function that calculates the expectation value of a given quantity (default is x) to be plugged into Reimann integrator.
    psi (function): wavefunction
    x (int/float): position wavefunction is evaluated at
    x_func (function): function that you are taking the expectation value of– x if not specified
    """
    wf = psi(x, **kwargs)                   # find the value of the wavefunction at x
    wfi = np.conjugate(wf)                  # get the complex conjugate of the wavefunction at x
    if x_func is not None:
        exp_val = wfi * x_func(x) * wf      # expectation value
    else: 
        exp_val = wfi * x * wf
    return exp_val

def expecVal2(psi, xmin, xmax, dx, x_func = None, **kwargs):
    """
    Function that calculates the expectation value of a given quantity (default is x) to be plugged into Reimann integrator.
    psi (function): wavefunction
    x (int/float): position wavefunction is evaluated at
    x_func (function): function that you are taking the expectation value of– x if not specified
    """
    def f(xmin, psi, x_func, **kwargs):
        wf = psi(xmin, **kwargs)                   # find the value of the wavefunction at x
        wfi = np.conjugate(wf)                  # get the complex conjugate of the wavefunction at x
        if x_func is not None:
            exp_val = wfi * x_func(xmin) * wf      # expectation value
        else: 
            exp_val = wfi * xmin * wf
        return exp_val
    exp_val_final = Reimann(f, xmin, xmax, dx, x_func = x_func, psi = psi, **kwargs)
    return exp_val_final


def fourierTr(xmin, xmax, dx, fx, **kwargs):
    """
    Fourier Transform function that performs the integral using the Reimann numerical solver
    """
    pmax = 1 / (xmax - xmin)
    pmin = -pmax
    c = 1 / np.sqrt(2 * np.pi * hbar)
    def f(xmin, pmin, fx, **kwargs):
        fp = c * np.exp((complex(0, -1) * pmin * xmin)/hbar) * fx(xmin, **kwargs)
        return fp 
    fp_final = Reimann(f, xmin, xmax, dx, fx = fx, pmin = pmin, **kwargs)
    return fp_final

a = 5
step = 0.1
test_int = fourierTr(0, a, step, psi_isw, n = 1, a = a)
test_int = expecVal2(psi_isw, 0, a, step, x_func = squared, n = 1, a = a)
print(test_int)


# I'm stuck so I started flipping through Griffiths chapter 2
# until I found something semi-interesting I could do

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

int_psiprod = Reimann(psitpsi, x1, x2, step, n1 = 3, n2 = 2, a = a, psi = psi_isw)
lhs = (dpsi_32 * psi_22) - (dpsi_31 * psi_21)
rhs = ((2*m)/hbar) * (E2 - E3) * int_psiprod



#print(E2)
#print(E3)
print(f'Left: {lhs}')
print(f'Right: {rhs}')

# To Do:
#   Improve comments– got kinda lazy there
#   Look into Griffiths problem further– do more psi states probably
#   Try part a with the Euler/RK4 calculators and see how the results match up with the integrator
