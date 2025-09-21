from dataclasses import dataclass, field, asdict
from typing import List, Literal
from collections.abc import Sequence


# CONFIGS
@dataclass
class ClusterConfig():
    """ Config to generate cluster scene """
    number_of_bubbles: int = 32
    """ The number of bubbles  """
    distance_variance: float = 1000.0   # 1 mm
    """ The variance of the pre-defined distance between bubbles in [μm] """
    minimum_distance: float = 150.0
    """ The minimum distance between bubbles in [μm] """
    distance_type: Literal["randomized", "equidistant"] = "randomized" 
    """ equidistant (mimimum distance), randomized (Gauss distribution) """
    bubble_size_range: List[float] = field(default_factory = lambda: [0.8, 5.0])
    """ The range of of equilibrium bubbles size in [μm] """
    bubble_size_type: Literal["fixed", "randomized", "equidistant"] = "randomized"     # fixed, randomized, equidistant
    """ The bubble size generation method: `fixed` `randomized` `equidistant`"""
    seed: int = 42
    """ The default seed for rng """
    plot_scene_to_html: bool = True
    """ If true a px.figure is rendered as a html """
    save_config_to_file: bool = True
    """ If true the cluster config is saved as npz files """
    save_as_mat: bool = True
    """ If true the config file is saved as *.mat file """


@dataclass
class GeneralParameters():
    """ Config for physical simulation parameters """
    P1: float = 1.0
    """ Pressure Amplitude 1 [bar] """
    P2: float = 25.0
    """ Frequency 1 [kHz] """
    P3: float = 0.0
    """ Pressure Amplitude 2 [bar] """
    P4: float = 0.0
    """ Frequency 2 [kHz] """
    P5: float = 0.0
    """ Phase Shift """
    P6: Sequence[float] = field(default_factory= lambda: [10.0e-6])
    """ Equilibrium Radii, placeholder -> overwritten later """
    P7: float = 1.0
    """ Ambient pressure [bar] """
    P8: float = 25.0
    """ Ambient tempearature [°C] """
    P9: float = 1.4
    """ Polytrophic exponenet """


@dataclass
class SolverConfig():
    """ Parameters for configure numerical soler """
    bubble_model: Literal["baseline", "jacobi_bruteforce", "jacobi_fitted", "jacobi_baseline",
                           "bicg", "bicgstab"] = "baseline"
    """
    Chose between different implementation versions
        - baseline          : The linear system is solved by numpy.linalg.solve
        - jacobi_bruteforce : The linear system is solved by Jacobi iteration with optimized omega; omega is searched via brute force
        - jacobi_fitted     : The linear system is solved by the matrix-free Jacobi iteration with optimized omega; omega is obtained as a function of max(x1)
        - jacobi_baseline   : The linear system is solved by the matrix-free Jacobi iteration with fixed omega; omega=1.0
        - bicg              : The linear system is solved by the matrix-free BiCG algorithm (accelerated using @numba.njit)
        - bicgstab          : The linear system is solved by the matrix-free BiCGSTAB algorithm (accelerated using @numba.njit)
    """
    jacobi_params: List[float] = field(default_factory = lambda: [0.5, 0.5, -0.04])
    """
    a, b, c --> omega_opt = a + b * exp(-c * rmax)
        - N32 -> [0.5, 0.5, -0.04]
        - N64 -> [0.4, 0.6, -0.06]
        - N128-> [0.3, 0.7, -0.07]
    """
    measure: Literal["runtime", "linalg", "simple"] = "simple"
    """
    Collect metrics
        - runtime : measure simulation time withot collecting metrics
        - linalg  : collect metrics related to the linalg solver (iterations, linalg error, omega_opt etc)
        - simple  : simple simulation
    """
    repeat_measure: int = 10
    """ If `measure` is `runtime`, the measure is repeated `repeat_measure` times """
    save_results: bool = True
    """ If true the results are saved to npz files """
    method: Literal["RK45", "Radau", "LSODA"] = "LSODA"
    """ Integration method  """
    tspan: float = 2.0
    """ Total simulation time (dimensionless)"""
    atol: float = 1.0e-9
    """ Integration absolute tolerance """
    rtol: float = 1.0e-9
    """ Integration realative tolerance """
    lin_atol: float = 1.0e-9
    """ Linsolve absolute tolerance (reziduum) ||b-Ax^(k)|| < lin_atol + ||x^(k) - x^(k-1)|| < lin_atol_x"""
    lin_rtol: float = 1.0e-9
    """ Linsolve relative tolerance (reziduum) ||b-Ax^(k)||/||b|| <lin_rtol + ||x^(k) - x^(k-1)||/||x^(k)|| < lin_rtol"""
    max_iter: int = 100
    """ Linsolve maximum number of iterations """
    plot_results: bool = False
    """ If the the R(t) curves are plotted after simulations """


