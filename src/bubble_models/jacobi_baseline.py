"""
Keller--Miksis bubble Model for N-Bubble Cluster in 3D fixed arrangement
This implementation uses a Matrix-free BiCGSTAB to solve the linalg system
"""

import numpy as np
import numba as nb

from . import register_ode

@nb.njit(cache=True, fastmath=True, inline='always')
def _acx(D1, D2, cp, v) -> np.ndarray:
    return D1 * (cp @ (D2 * v)) #+ v


@nb.njit(cache=True, fastmath=True)
def _jacobi_iter(D1, D2, cp, b, omega, atol, rtol, max_iter):
    """
    Solves (A + I)x = b using a relaxed Jacobi iteration:
    diag(A) = 0!!
    """
    x = b.copy()
    bnorm = np.linalg.norm(b)
    tol = max(atol, rtol * bnorm)
    rnorm = np.inf
    step_norm = np.inf
    for k in range(1, max_iter+1):
        Ax = _acx(D1, D2, cp, x)
        # r = b - (A + I) x = b - Ax - x
        r = b - (Ax + x)
        rnorm = np.linalg.norm(r)
        if rnorm <= tol:
            return x, k, True, rnorm, step_norm
        x_new = omega * (b - Ax) + (1.0 - omega) * x
        step_norm = np.linalg.norm(x_new - x)
        if step_norm < max(atol, rtol * np.linalg.norm(x_new)):
            return x_new, k, True, rnorm, step_norm
        x = x_new
    return x, k, False, rnorm, step_norm


@register_ode("jacobi_baseline")
@nb.njit(cache=True, fastmath=True)
def _ode_function(t, x, up, cp, d, num_bubbles, atol, rtol, max_iter, metrics_buffer, *args):
    """
    Dimensionless Keller--Miksis equation for N-bubble coupled system with fixed positions
    Linalg system is solved by a matrix-free jacobi algorith useing a fitted function to calculate omega_opt
    Parameters \n
    ---------
    t   : Dimnesionless time
    x   : State variables (R, dR)
    up  : Unit-scope control parameters
    cp  : Constant Coupling Matrix
    d   : Distance Matrix (event)
    num_bubbles   : Number of Bubbles
    atol: absolute tolerance of Jacobi iteration
    rtol: relative tolernace of Jacobi iteration 
    max_iter: maximum number of iteration per function evaluation
    metrics_buffer: NDarray to keep actual metrics (outside jit function)
        0 - timestep
        1 - rmax
        2 - omega_opt
        3 - max_iter
        4 - linalg error
    Returns \n
    ---------
    dx  : Time derivative of state variables dx = (A(x, t)+I)^-1 * f(x,t)
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
    rhs = rD * (N - 2.0 * cp @ (x1 * x2**2))

    rmax = np.max(x1)
    dx_sol, iter, _, rnorm, step_norm = _jacobi_iter(rD, x1**2, cp, rhs, 1.0, atol, rtol, max_iter)
    dx[num_bubbles:] = dx_sol

    metrics_buffer[0] = t
    metrics_buffer[1] = np.max(x1)
    metrics_buffer[2] = 1.0              # This method does not use a relaxation factor
    metrics_buffer[3] = iter
    metrics_buffer[4] = rnorm
    metrics_buffer[5] = step_norm

    return dx