import os
import tyro
import time
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp

from utils import CLUSTER_DATA, TIME_SERIES_DATA, METRICS_DATA, RUNTIME_DATA
from utils import make_file_name, make_run_name
from utils import compute_equation_constants
from utils import ClusterConfig, GeneralParameters, SolverConfig
from bubble_size_and_spatial_ditribution_generator import generate_scene

# Load Available Models
from bubble_models import collision_event
from bubble_models import baseline, bicg, bicgstab, jacobi_bruteforce, jacobi_fitted, jacobi_baseline
from bubble_models import load_ode, available_models

# Metrics
from utils.metrics import collect_metrics, metrics_run

import matplotlib.pyplot as plt

@dataclass
class Args:
    general : GeneralParameters = field(default_factory=GeneralParameters)
    cluster : ClusterConfig = field(default_factory=ClusterConfig)
    solver  : SolverConfig = field(default_factory=SolverConfig)


def load_scene(args):
    file_name = make_file_name(args.cluster,
            exclude_keys=["plot_scene_to_html", "save_config_to_file", "save_as_mat"])
    
    if not os.path.exists(CLUSTER_DATA / f"{file_name}.npz"):
        print("Generate a new scene with the actual properties")
        generate_scene(args.cluster)
    try:
        print("Loading scene from file.")
        scene_data = np.load(CLUSTER_DATA / f"{file_name}.npz")
        bubble_sizes     = scene_data["bubble_sizes"]
        bubble_positions = scene_data["bubble_positions"]
        distance_matrix  = scene_data["distance_matrix"]
        coupling_matrix  = scene_data["coupling_matrix"]
        minimum_distance = scene_data["min_distance"]

    except KeyError as e:
        print(e)

    return bubble_sizes, bubble_positions, distance_matrix, coupling_matrix, minimum_distance



def solve_scene(args, unit_parameters, coupling_matrix, distance_matrix):
    metrics_buffer = np.zeros(6, dtype=np.float64)          # Must be generated!!

    if args.solver.measure in ["runtime", "simple"]:
        # Use the njit compiled ode fun
        ode_fun = load_ode(args.solver.bubble_model)
        metrics_enabled = False
    
    elif args.solver.measure == "linalg":
        # Wrap the ode_fun to keep metrics
        #core_ode_fun = load_ode(args.solver.bubble_model)
        def wrapmeplease(odefn):
            def ode_wrapped(t, x, *args):
                dx = odefn(t, x, *args)
                collect_metrics(metrics_buffer)
                return dx
            return ode_wrapped
        
        ode_fun = wrapmeplease(load_ode(args.solver.bubble_model))
        metrics_enabled = True

    # --- Simple Run and Metric Collection ---
    if args.solver.measure in ["linalg", "simple"]:
        with metrics_run(metrics_enabled) as metrics:
            try:
                start = time.time()
                results = solve_ivp(
                    fun=ode_fun,
                    t_span = [0.0, args.solver.tspan],
                    y0 = np.hstack((np.ones(args.cluster.number_of_bubbles), np.zeros(args.cluster.number_of_bubbles))),
                    args = (unit_parameters, coupling_matrix, distance_matrix, args.cluster.number_of_bubbles, \
                        args.solver.lin_atol, args.solver.lin_rtol, args.solver.max_iter, metrics_buffer, args.solver.jacobi_params ),
                    method=args.solver.method,
                    atol=args.solver.atol,
                    rtol=args.solver.rtol,
                    min_step=1e-16,
                    events=collision_event)
                
                end = time.time()
                simulation_time = end - start
                print(results.message)
                print(f"The total simulation time was: {simulation_time:.4f} seconds")
                print(f"stats: nfev={results.nfev}, njev={getattr(results, 'njev', 'NA')}, nlu={getattr(results, 'nlu', 'NA')}")

                if args.solver.save_results:
                    root_folder = make_file_name(args.cluster,
                        exclude_keys=["plot_scene_to_html", "save_config_to_file", "save_as_mat"])
                    run_name = make_run_name(args.general)
                    
                    os.makedirs(TIME_SERIES_DATA / root_folder, exist_ok=True)
                    os.makedirs(METRICS_DATA / root_folder, exist_ok=True)

                    np.savez_compressed(
                        TIME_SERIES_DATA / root_folder / f"{run_name}_{args.solver.bubble_model}.npz",
                        time = results.t,
                        radii = results.y[:args.cluster.number_of_bubbles].T,
                        radii_micron  = results.y[:args.cluster.number_of_bubbles].T * args.general.P6 * 1e6
                    )

                    if metrics_enabled:
                        np.savez_compressed(
                            METRICS_DATA / root_folder / f"{run_name}_{args.solver.bubble_model}.npz",
                            time  = metrics.time,
                            omega = metrics.omega_opt,
                            rmax  = metrics.rmax,
                            iters = metrics.iters,
                            linalg_error = metrics.linalg_error,
                            last_stepsize = metrics.last_stepsize,
                            atol = args.solver.atol,
                            rtol = args.solver.rtol,
                            lin_atol = args.solver.lin_atol,
                            lin_rtol = args.solver.lin_rtol,
                            simulation_time = simulation_time
                        )
            
                    if args.solver.plot_results:
                        plt.figure(1)
                        plt.plot(results.t, results.y[:args.cluster.number_of_bubbles].T)
                        plt.show()

            except KeyboardInterrupt:
                print("Simulation terminated by the user")

        
    # --- Computation Time Measurement ---
    elif args.solver.measure == "runtime":
        try:
            # Run a short warm-up
            results = solve_ivp(
                    fun=ode_fun,
                    t_span = [0.0, 0.25],
                    y0 = np.hstack((np.ones(args.cluster.number_of_bubbles), np.zeros(args.cluster.number_of_bubbles))),
                    args = (unit_parameters, coupling_matrix, distance_matrix, args.cluster.number_of_bubbles, \
                        args.solver.lin_atol, args.solver.lin_rtol, args.solver.max_iter, metrics_buffer, args.solver.jacobi_params ),
                    method=args.solver.method,
                    atol=args.solver.atol,
                    rtol=args.solver.rtol,
                    min_step=1e-16,
                    events=collision_event)
            print("Warm-up simulation is ended")
            print(results.message)
            print("-----------------------")
            # -- Time Measurement --
            sim_time = []
            for repeat in range(1,args.solver.repeat_measure+1):
                start = time.perf_counter()
                results = solve_ivp(
                    fun=ode_fun,
                    t_span = [0.0,args.solver.tspan],
                    y0 = np.hstack((np.ones(args.cluster.number_of_bubbles), np.zeros(args.cluster.number_of_bubbles))),
                    args = (unit_parameters, coupling_matrix, distance_matrix, args.cluster.number_of_bubbles, \
                        args.solver.lin_atol, args.solver.lin_rtol, args.solver.max_iter, metrics_buffer, args.solver.jacobi_params ),
                    method=args.solver.method,
                    atol=args.solver.atol,
                    rtol=args.solver.rtol,
                    min_step=1e-18,
                    events=collision_event)
                end = time.perf_counter()
                simulation_time = end - start
                sim_time.append(simulation_time)
                print(f"{repeat}/{args.solver.repeat_measure} simulation is ended.")
                print(results.message)
                print(f"The total simulation time was: {simulation_time:.4f} seconds")

            if args.solver.save_results:
                root_folder = make_file_name(args.cluster,
                        exclude_keys=["plot_scene_to_html", "save_config_to_file", "save_as_mat"])
                run_name = make_run_name(args.general)

                os.makedirs(RUNTIME_DATA / root_folder, exist_ok=True)
                np.savez_compressed(
                    RUNTIME_DATA / root_folder / f"{run_name}_{args.solver.bubble_model}.npz",
                    sim_time  = sim_time )


        except KeyboardInterrupt:
            print("Simulation terminated by the user")




if __name__ == "__main__":

    args = tyro.cli(Args)

    # --- LOAD SCENE ---
    bubble_sizes,    \
    bubble_positions,\
    distance_matrix, \
    coupling_matrix, \
    minimum_distance = load_scene(args)

    
    # --- CALCULATE UNIT-SCOPE CONSANTS --
    args.general.P6 = bubble_sizes.ravel() * 1e-6      # Set bubble sizes as row-vector (convert micron to m!!)
    unit_parameters = compute_equation_constants(
                    args.general, 
                    args.cluster.number_of_bubbles)
    

    # --- LAUNCH THE CURRENT SIMULATION --
    solve_scene(args, unit_parameters, coupling_matrix, distance_matrix)


