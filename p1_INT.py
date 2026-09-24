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

hbar = const.hbar           # Reduced Planck Constant

def psi_isw(x, a, n, **kwargs):
    """
    Position wavefunction for the infinite square well
    x (int/float): position wavefunction is evaluated at
    a (int/float): width of the well
    n (int): energy level 
    """
    psi = np.sqrt(2 / a) * np.sin((n * np.pi * x) / a)
    return psi

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
    exp_val_final = Reimann(f, xmin, xmax, dx, x_func = x_func, psi = psi_isw, **kwargs)
    return exp_val_final


def fourierTr(x, p, fx):
    pass



a = 5
step = 0.1
test_int = expecVal2(psi_isw, 0, a, step, x_func = squared, n = 1, a = a)
#test_int = Reimann(expecVal, (a * -1), a, step, psi = psi_isw, x_func = squared, A = 0, n = 1, a = a)
print(test_int)