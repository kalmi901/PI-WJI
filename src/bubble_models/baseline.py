"""
Keller--Miksis bubble Model for N-Bubble Cluster in 3D fixed arrangement
This implementation serves as a baseline

"""
import numpy as np
import numba as nb

from . import register_ode

@register_ode("baseline")
@nb.njit(cache=True, fastmath=True)
def _ode_function(t, x, up, cp, d, num_bubbles, *args):
    """
    Dimensionless Keller--Miksis equation for N-bubble coupled system with fixed positions
    Linalg system is solved by numpy.linalg.solve
    Parameters \n
    ---------
    t   : Dimnesionless time
    x   : State variables (R, dR)
    up  : Unit-scope control parameters
    cp  : Constant Coupling Matrix
    d   : Distance Matrix (event)
    num_bubbles   : Number of Bubbles
    Returns \n
    ---------
    dx  : Time derivative of state variables dx = A(x, t)^-1 * f(x,t)
    """

    x1 = x[:num_bubbles]
    x2 = x[num_bubbles:]

    dx = np.zeros_like(x)

    # --- Uncoupled part ---
    rx1 = 1.0 / x1
    
    N = (up[0]+up[1]*x2)*rx1**up[10] - up[2]*(1.0+up[9]*x2) - up[3]*rx1 - up[4]*x2*rx1 - (1.5-0.5*up[9]*x2)*x2*x2 - ( up[5]*np.sin(2.0*np.pi*t) + up[6]*np.sin(2.0*up[11]*np.pi*t+up[12]) ) * (1.0+up[9]*x2) - x1*( up[7]*np.cos(2.0*np.pi*t) + up[8]*np.cos(2.0*up[11]*np.pi*t+up[12]) )
    D = x1 - up[9]*x1*x2 + up[4]*up[9]
    rD = 1.0/D

    dx[:num_bubbles] = x2
    #dx[num_bubbles:] = N * rD
    
    # Implicit Coupling matrix
    A = rD[:, np.newaxis] * (cp * x1[np.newaxis, :]**2)      # Implicit coupling term with broadcast
    np.fill_diagonal(A, 1.0)                                 # + EYE inplace

    # Explicit Coupling
    rhs = rD * (N - 2.0 * cp @ (x1 * x2**2))
    dx[num_bubbles:] = np.linalg.solve(A, rhs)

    return dx



