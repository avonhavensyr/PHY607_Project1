"""
PHYSICS 607 PROJECT 1

METHODS
"""

def Euler(m, v, x, xi, dt):
    """
    Function that evaluates a new value of a function and its derivative
    based previous input values and returns the new state as a vector using
    the symplectic Euler method
    """
    # going to use symplectic method
    # starting with the numbers as singular values rather than as arrays like
    # previous Euler solvers
    dv = f(x, xi, m)
    v_new = v + (dv * dt)       # velocity from Euler method
    x_new = x + (v_new * dt)    # position from Euler method
    return np.array([x_new, v_new])

def RK4(mi, mj, vi, vj, xi, xj, dt):
    """
    Function that evaluates a new value of a function and its derivative
    based previous input values and returns the new state as a vector using
    the fourth-order Runge-Kutta method
    """
    k1xi = vi
    k1vi = dt * f(xi, xj, mi)
    k1xj = vj
    k1vj = dt * f(xj, xi, mj)

    k2xi = vi + ((dt/2) * k1vi)
    k2vi = f((xi + ((dt/2) * k1xi)), (xj + ((dt/2) * k1xj)), mi)
    k2xj = vj + ((dt/2) * k1vj)
    k2vj = f((xj + ((dt/2) * k1xj)), (xi + ((dt/2) * k1xi)), mj)

    k3xi = vi + ((dt/2) * k2vi)
    k3vi = f((xi + ((dt/2) * k2xi)),(xj + ((dt/2) * k2xj)), mi)
    k3xj = vj + ((dt/2) * k2vj)
    k3vj = f((xj + ((dt/2) * k2xj)),(xi + ((dt/2) * k2xi)), mj)

    k4xi = vi + ((dt/2) * k3vi)
    k4vi = f((xi + ((dt/2) * k3xi)),(xj + ((dt/2) * k3xj)), mi)
    k4xj = vj + ((dt/2) * k3vj)
    k4vj = f((xj + ((dt/2) * k3xj)),(xi + ((dt/2) * k3xi)), mj)

    xi_new = xi + ((dt/6) * (k1xi + (2*k2xi) + (2*k3xi) + k4xi))
    vi_new = vi + ((dt/6) * (k1vi + (2*k2vi) + (2*k3vi) + k4vi))

    xj_new = xj + ((dt/6) * (k1xj + (2*k2xj) + (2*k3xj) + k4xj))
    vj_new = vj + ((dt/6) * (k1vj + (2*k2vj) + (2*k3vj) + k4vj))

    return (np.array([xi_new, vi_new]), np.array([xj_new, vj_new]))