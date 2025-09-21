import numpy as np
import re
from numpy.typing import NDArray
from dataclasses import asdict
from typing import List
from utils import ClusterConfig, GeneralParameters

# --- MATERIAL PROPERTIES ---
PV  = 3.166775638952003e+03         # Vapor Pressure [Pa]
RHO = 9.970639504998557e+02         # Liquid Density [kg / m**3]
ST  = 0.071977583160056             # Surface Tension [N / m]
VIS = 8.902125058209557e-04         # Liquid Dynamic Viscosity [Pa s]
CL  = 1.497251785455527e+03         # Liquid Sound Speed




def generate_bubble_size(
        bubble_size_range: List[float], 
        number_of_bubbles: int, 
        bubble_size_type: str,
        seed: int) -> NDArray[np.float64]:
    
    match bubble_size_type:
        case "fixed":
            return np.full((number_of_bubbles, 1), bubble_size_range[0], dtype=np.float64)  
        case "randomized":
            rng = np.random.default_rng(seed)
            return bubble_size_range[0] + rng.random((number_of_bubbles, 1), dtype=np.float64) * (bubble_size_range[1] - bubble_size_range[0])
        case "equidistant":
            return np.linspace(bubble_size_range[0], bubble_size_range[1], number_of_bubbles, dtype=np.float64).reshape(-1, 1)
        case _:
            raise RuntimeError(f"{bubble_size_type} is incorrect. Supported bubble_size_types are `fixed`, `randomized` and `equidistant`")

def generate_bubble_position(
        distance_variance: float,
        minimum_distance: float,
        number_of_bubbles: int, 
        distance_type: str,
        seed: int):
        
    positions = np.zeros((number_of_bubbles, 3), dtype=np.float64)
    match distance_type:
        
        case "equidistant":
           positions[:, 0] = np.arange(number_of_bubbles) * minimum_distance
           return positions
        
        case "randomized":
            rng = np.random.default_rng(seed)
            for i in range(number_of_bubbles):
                while True:
                    candidate_position = rng.normal(loc=0.0, scale=distance_variance, size=3)
                    if i == 0:
                        positions[0,:] = candidate_position
                        break # break the while ->
                    else:
                        # Check distance
                        if np.all(np.linalg.norm(positions[:i] - candidate_position, axis=1) >= minimum_distance):
                            positions[i, :] = candidate_position
                            break
            return positions
        
        case _:
            raise RuntimeError(f"{distance_type} is incorrect. Supported distance_type are `randomized` and `equidistant`")

def compute_equation_constants(parameters: GeneralParameters,
                               number_of_bubbles: int = 1) -> NDArray[np.float64]:
    
    if len(parameters.P6) != number_of_bubbles:
        raise RuntimeError("The number of bubbles sizes is not correct!")
    
    Pinf = parameters.P7 * 1e5
    Pa1  = parameters.P1 * 1e5
    Pa2  = parameters.P3 * 1e5

    w1 = 2*np.pi * parameters.P2 * 1000      # omega1
    w2 = 2*np.pi * parameters.P4 * 1000      # omega2

    up = np.zeros((17, number_of_bubbles), dtype=np.float64)

    up[0]  = (2*ST/parameters.P6 + Pinf - PV) * (2.0*np.pi/parameters.P6/w1)**2 / RHO
    up[1]  = (1-3.0*parameters.P9) * (2*ST/parameters.P6 + Pinf - PV) * (2.0*np.pi/parameters.P6/w1) / CL/RHO
    up[2]  = (Pinf - PV) * (2.0*np.pi/parameters.P6/w1)**2 / RHO
    up[3]  = (2*ST/parameters.P6/RHO) * (2.0*np.pi/parameters.P6/w1)**2
    up[4]  = (4*VIS/RHO/parameters.P6**2) * (2.0*np.pi/w1)
    up[5]  = Pa1 * (2.0*np.pi/parameters.P6/w1)**2 / RHO
    up[6]  = Pa2 * (2.0*np.pi/parameters.P6/w1)**2 / RHO
    up[7]  = (parameters.P6*w1*Pa1/RHO/CL) * (2.0*np.pi/parameters.P6/w1)**2
    up[8]  = (parameters.P6*w2*Pa2/RHO/CL) * (2.0*np.pi/parameters.P6/w1)**2
    up[9]  = parameters.P6*w1/(2.0*np.pi)/CL
    up[10] = 3.0*parameters.P9
    up[11] = parameters.P4/parameters.P2
    up[12] = parameters.P5

    up[13] = 2.0*np.pi/w1       # tref
    up[14] = parameters.P6      # Rref

    up[15] = w1                 # omega1
    up[16] = w2                 # omega2

    return up


def compute_coupling_matrix(
        distance_matrix: NDArray[np.float64],
        bubble_sizes: NDArray[np.float64]) -> NDArray[np.float64]:
    
    if distance_matrix.shape[0] != len(bubble_sizes):
        raise RuntimeError("The number of bubbles sizes is not correct!")

    with np.errstate(divide='ignore'):       # Diagonals are 1/0
        r_distance = 1.0 / distance_matrix 
        np.fill_diagonal(r_distance, 0.0)    # No Self coupling
        
    R0_i = bubble_sizes                 # (num_bubbles, 1)
    R0_j = bubble_sizes.transpose()     # (1, num_bubbles)

    coupling_matrix = (R0_j**3) / (R0_i **2) * r_distance
    #print(coupling_matrix.data.c_contiguous)
    return coupling_matrix


FILENAME_MAP = {
    "number_of_bubbles": "N",
    "distance_variance": "DVar",
    "minimum_distance" : "dMin" ,
    "bubble_size_range": "RE",
    "distance_type"    : "dType",
    "bubble_size_type" : "RType",
    "seed"             : "S"
}

def make_file_name(
        args: ClusterConfig, 
        prefix: str = "", 
        suffix: str = "", 
        exclude_keys: List[str] | None = None):
    """ Generate standardized filename from args dataclasses """
    
    exclude_keys = exclude_keys if exclude_keys is not None else []
    args_dict = {FILENAME_MAP[key]: value for key, value in asdict(args).items() if key not in exclude_keys}

    parts = []
    for k, v in args_dict.items():
        if isinstance(v, float):
            parts.append(f"{k}-{v:.2f}")
        elif isinstance(v, int):
            parts.append(f"{k}-{v:d}")
        else:
            parts.append(f"{k}-{v}")
    safe_parts = [re.sub(r"[^A-Za-z0-9_.-]", "-", p) for p in parts]        # AI-assisted snippet 

    filename = "-".join(filter(None, [prefix] + safe_parts + [suffix]))
    return filename


def make_run_name(
        args: GeneralParameters,
        prefix: str = "",
        suffix: str = ""):
    """ Generate standardized filename from General parameters """

    exclude_keys = ["P5", "P6", "P7", "P8", "P9"]     # Constants in the present study
    args_dict = {key: value for key, value in asdict(args).items() if key not in exclude_keys}
    parts = []
    for k, v in args_dict.items():
        parts.append(f"{k}-{v:.2f}")
    filename = "-".join(filter(None, [prefix] + parts + [suffix]))
    return filename