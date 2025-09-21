"""
Keller--Miksis bubble Model for N-Bubble Cluster in 3D fixed arrangement
This implementation uses a Matrix-free BiCG to solve the linalg system
"""

import numpy as np
import numba as nb

from . import register_ode

@nb.njit(cache=True, fastmath=True, inline='always')
def _ax(D1, D2, cp, v) -> np.ndarray:
    return D1 * (cp @ (D2 * v)) + v

@nb.njit(cache=True, fastmath=True, inline='always')
def _atx(D1, D2, cp, v) -> np.ndarray:
    return D2 * (cp.T @ (D1 * v)) + v

@nb.njit(cache=True, fastmath=True)
def _bicg(D1, D2, cp, b, atol, rtol, max_iter):
    x = b.copy()
    r = b - _ax(D1, D2, cp, x)
    bnorm = np.linalg.norm(b)
    tol = max(atol, rtol * bnorm)
    rnorm = np.linalg.norm(r)
    if rnorm <= max(atol, rtol * bnorm):
        return x, 0, True, rnorm, 0.0

    rho_old = 1.0
    rt = r.copy()
    p  = np.zeros_like(b)
    pt = np.zeros_like(b)
    eps = 1.0e-30

    for k in range(1, max_iter+1):
        rho = np.dot(r, rt)
        if np.abs(rho) < eps:
            # BiCG method failure
            return x, k, False, rnorm, 0.0
        if k == 1:
            p[:]  = r
            pt[:] = rt
        else:
            beta = rho / rho_old
            p  = r  + beta * p
            pt = rt + beta * pt

        q = _ax(D1, D2, cp, p)
        qt = _atx(D1, D2, cp, pt)

        denom = np.dot(pt, q)
        if np.abs(denom) < eps:
            # BiCG method failure
            return x, k, False, rnorm, 0.0

        alpha = rho / denom
        step_size = alpha * p
        step_norm = np.linalg.norm(step_size)
        x += step_size
        r -= alpha * q
        rt-= alpha * qt
        rnorm = np.linalg.norm(r)
        if rnorm <= tol or np.linalg.norm(step_size) < max(atol, rtol * np.linalg.norm(x)):
            return x, k, True, rnorm, step_norm
        
        rho_old = rho
        
    return x, k, False, rnorm, step_norm


@register_ode("bicg")
@nb.njit(cache=True, fastmath=True)
def _ode_function(t, x, up, cp, d, num_bubbles, atol, rtol, max_iter, metrics_buffer, *args):
    """
    Dimensionless Keller--Miksis equation for N-bubble coupled system with fixed positions
    Linalg system is solved by a matrix-free bicg algorithm
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
 
    # Implicit Coupling matrix
    #A = rD[:, np.newaxis] * (cp * x1[np.newaxis, :]**2)      # Implicit coupling term with broadcast
    #np.fill_diagonal(A, 1.0)                                 # ADD DIAGONALS HERE
    # Explicit Coupling
    rhs = rD * (N - 2.0 * cp @ (x1 * x2**2))
    #dx[num_bubbles:] = np.linalg.solve(A, rhs)

    dx_sol, iter, _, rnorm, step_size = _bicg(rD, x1**2, cp, rhs, atol, rtol, max_iter)
    dx[num_bubbles:] = dx_sol

    metrics_buffer[0] = t
    metrics_buffer[1] = np.max(x1)
    metrics_buffer[2] = -1              # This method does not use a relaxation factor
    metrics_buffer[3] = iter
    metrics_buffer[4] = rnorm
    metrics_buffer[5] = step_size

    return dx