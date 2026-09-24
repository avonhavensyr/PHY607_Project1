import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const


"""
PROJECT I
"""

"""
Numerical Integrator: Quantum Harmonic Oscillator
"""
# Note for instructor: I just got a Macbook from the school and discovered
# the "option" key– all em-dash characters present in code comments are
# manually typed by me hitting the option key with ZERO assistance from AI

# going to start with a simple Reimann sum– I hate hate hate doing
# the easiest option provided, but I need to make sure I can submit
# complete by Tuesday
# Reduced Planck Constant, SI units
hbar = const.hbar

def psi0(m, omega):
    c = ((m * omega) / (np.pi * hbar)) ** (1/4)     # Ground-state co-efficient
    a = ((m * omega) / (2 * hbar))                  # Factor in the exponent– little a 

def f_test(x):
    return x**2

def Reimann(f, x0, dx, xmax, A=0):
    """
    f (function): Function being integrated
    x0 (int/float): Value being integrated over
    dx (int/floar): Step-size
    xmax (int/float): Upper integration bound
    """
    h = f(x0)           # Find the height
    A += h * dx          # Calculate the area of the Reimann sum rectangle
    if x0 > xmax:
        return A
    else:
        return Reimann(f, x0 + dx, dx, xmax, A)

l = 1
u = 10
step = 0.00909
test_int = Reimann(f_test, l, step, u)
print(test_int)
nsteps = (u - l)/ step
print(f'Steps: {nsteps}')
# Test implies that I'm getting a recursion depth of ≈1000