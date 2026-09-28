"""
PHYSICS 607 PROJECT 1

METHODS
"""

# ----------------------- LIBRARY DEPENDENCIES + CONSTANT -----------------------

import matplotlib as plt
import numpy as np

hbar = 1

# ----------------------- ODE SOLVERS -----------------------

def Euler(f, v, x, dt, **kwargs):
    """
    Function that evaluates a new value of a function and its derivative
    based previous input values and returns the new state as a vector using
    the symplectic Euler method

    Keyword Arguments:
    f (function): function that returns the derivative of v-- must take x as first argument
    v (int/float): initial/current velocity value
    x (int/float): initial/current position value
    dt (int/float): time interval
    """

    dv = f(x, **kwargs)
    v_new = v + (dv * dt)       # velocity from Euler method
    x_new = x + (v_new * dt)    
    return np.array([x_new, v_new])

def RK4(f, mi, mj, vi, vj, xi, xj, dt):
    """
    Function that evaluates a new value of a function and its derivative
    based previous input values and returns the new state as a vector using
    the fourth-order Runge-Kutta method FOR THE COUPLED HARMONIC OSCILLATOR
    Note: not generalized to all problems

    Keyword Arguments:
    f (function): function that returns the derivative of v-- must take x as first argument
    mi (int/float): mass of mass i
    mj (int/float): mass of mass j
    vi (int/float): initial/current velocity value of mass i
    vj (int/float): initial/current velocity value of mass j
    xi (int/float): initial/current position value of mass i
    xj (int/float): initial/current position value of mass j
    dt (int/float): time interval
    """
    k1xi = vi
    k1vi = f(xi, xj, mi)
    k1xj = vj
    k1vj = f(xj, xi, mj)

    k2xi = (vi + ((dt/2) * k1vi))
    k2vi = (f((xi + ((dt/2) * k1xi)), (xj + ((dt/2) * k1xj)), mi))
    k2xj = (vj + ((dt/2) * k1vj))
    k2vj = (f((xj + ((dt/2) * k1xj)), (xi + ((dt/2) * k1xi)), mj))

    k3xi = (vi + ((dt/2) * k2vi))
    k3vi = (f((xi + ((dt/2) * k2xi)),(xj + ((dt/2) * k2xj)), mi))
    k3xj = (vj + ((dt/2) * k2vj))
    k3vj = (f((xj + ((dt/2) * k2xj)),(xi + ((dt/2) * k2xi)), mj))

    k4xi = (vi + ((dt) * k3vi))
    k4vi = (f((xi + ((dt) * k3xi)),(xj + ((dt/2) * k3xj)), mi))
    k4xj = (vj + ((dt) * k3vj))
    k4vj = (f((xj + ((dt) * k3xj)),(xi + ((dt/2) * k3xi)), mj))

    xi_new = xi + ((dt/6) * (k1xi + (2*k2xi) + (2*k3xi) + k4xi))
    vi_new = vi + ((dt/6) * (k1vi + (2*k2vi) + (2*k3vi) + k4vi))

    xj_new = xj + ((dt/6) * (k1xj + (2*k2xj) + (2*k3xj) + k4xj))
    vj_new = vj + ((dt/6) * (k1vj + (2*k2vj) + (2*k3vj) + k4vj))

    return (np.array([xi_new, vi_new]), np.array([xj_new, vj_new]))

# ----------------------- NUMERICAL INTEGRATORS -----------------------

def Reimann(f, xmin, xmax, dx, A=0, **kwargs):
    """
    Function that numerically integrates a given function, f, using the Reimann sum method

    Keywork Arguments:
    f (function): Function being integrated
    xmin (int/float): Value being integrated over
    xmax (int/float): Upper integration bound
    dx (int/floar): Step-size
    A (int): starting area under curve-- default 0
    """
    h = f(xmin, **kwargs)           # Find the height
    A += h * dx                     # Calculate the area of the Reimann sum rectangle
    if xmin > xmax:
        return A
    else:
        return Reimann(f, xmin + dx, xmax, dx, A, **kwargs)


def fourierTr(xmin, xmax, dx, fx, **kwargs):
    """
    Fourier Transform function that performs the integral using the Reimann numerical solver

    xmin (int/float): initial position wavefunction is evaluated at
    xmax (int/float): final position wavefunction is evaluated at
    dx (int/float): change in x between steps

    """
    pmax = 1 / (xmax - xmin)
    pmin = -pmax
    c = 1 / np.sqrt(2 * np.pi * hbar)
    def f(xmin, pmin, fx, **kwargs):
        fp = c * np.exp((complex(0, -1) * pmin * xmin)/hbar) * fx(xmin, **kwargs)
        return fp 
    fp_final = Reimann(f, xmin, xmax, dx, fx = fx, pmin = pmin, **kwargs)
    return fp_final