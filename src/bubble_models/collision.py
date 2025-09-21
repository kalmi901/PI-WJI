"""
Event function to terminate ODE solvers at bubble collisison
"""
import numpy as np
import numba as nb

COLLISION_THRESHOLD = 0.10        # Micron

@nb.njit
def collision_event(t, x, up, _, d, num_bubbles, *args):
    x1 = x[:num_bubbles] * up[14] * 1e6          # Radius in micron
    min_val = np.inf
    # TRIU -> Condition Check -> (Ri + Rj) + eps > d(i,j)
    for i in range(num_bubbles):
        for j in range(num_bubbles):
            if i != j:
                min_delta = (x1[i] + x1[j])
                #min_delta = (x1[i] + x1[j]) +  COLLISION_THRESHOLD 
                val = d[i,j] * (1 - COLLISION_THRESHOLD ) - min_delta
                if val < min_val:
                    min_val = val
    return min_val
collision_event.terminal  = True
collision_event.direction = 0