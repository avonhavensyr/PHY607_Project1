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

def Reimann(f, x, xmax, dx, A, **kwargs):
    """
    f (function): Function being integrated
    xmin (int/float): Value being integrated over
    xmax (int/float): Upper integration bound
    dx (int/floar): Step-size
    A (int): starting area under curve
    """
    h = f(x, **kwargs)           # Find the height
    A += h * dx                     # Calculate the area of the Reimann sum rectangle
    if x > xmax:
        return A
    else:
        return Reimann(f, x + dx, xmax, dx, A, **kwargs)

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
def expecX(x, psi, x_func = None, **kwargs):
    """
    Helper function that helps expectation value
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

def expecValue():
    pass

a = 5
step = 0.1
test_int = Reimann(expecX, (a * -1), a, step, psi = psi_isw, x_func = squared, A = 0, n = 1, a = a)
print(test_int)