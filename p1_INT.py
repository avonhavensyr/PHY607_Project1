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
hbar = const.hbar

def psi0(x, a, n):
    """
    Ground-state wavefunction for the quantum harmonic oscillator
    at specified positon, x
    x (int/float): position wavefunction is evaluated at
    a (int/float): width of the well
    n (int): energy level 
    """
    psi = np.sqrt(2 / a) * np.sin((n * np.pi * x) / a)
    return psi

def f_test(x):
    return x**2

def Reimann(f, x0, dx, xmax, A, *args):
    """
    f (function): Function being integrated
    x0 (int/float): Value being integrated over
    dx (int/floar): Step-size
    xmax (int/float): Upper integration bound
    """
    h = f(x0, *args)           # Find the height
    A += h * dx          # Calculate the area of the Reimann sum rectangle
    if x0 > xmax:
        return A
    else:
        return Reimann(f, x0 + dx, dx, xmax, A, *args)

l = 1
u = 10
step = 0.00909
test_int = Reimann(f_test, l, step, u)
print(test_int)
nsteps = (u - l)/ step
print(f'Steps: {nsteps}')
# Test implies that I'm getting a recursion depth of ≈1000