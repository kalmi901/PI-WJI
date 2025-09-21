# PI-WJI
Physics-informed weighted Jacobi iteration for coupled bubble systems

## Description ##

A memory-efficient iterative technique is proposed for solving the implicit linear system arising in the coupled bubble dynamical model referred to as **physics-informed weighted Jacobi iteration (PI-WJI)**. The proposed method is based on the weighted Jacobi iteration, in which the optimal relaxation parameter $`\omega_{opt}`$ is derived from the instantaneous bubble radii, improving stability and convergence speed. Furthermore, a coefficient matrix decomposition scheme tailored to bubble clusters is introduced, enabling efficient matrix-free implementations of stationary and Krylov subspace iterative solvers.

### Matrix Decomposition ###
The dimensionless ODE system describing the dynamics of an N-bubble system is written as

```math
\begin{align*}
\dot{x}_{1,i}&=x_{2,i}, \\
\sum_{j=1}^N{B_{i,j}\dot{x}_{2,j}}&=b_i
\end{align*},
```

where $`x_{1,i}=R_i/R_{0,i}`$ is the dimensionless bubble radius, $`b_i`$ is the right-hand side of the ODE and $`B_{i,j}`$ is the instantaneous coupling matrix. Exploiting the special structure of the coupling matrix, the following decomposition is applied

```math
B_{i,j}=D_{1,i,j}\cdot A_{i,j} \cdot D_{2,i,j} + I_{i,j},
```

where $`D_{1,i,j}`$ and $`D_{2,i,j}`$ are diagonal matrices, $`A_{i,j}`$ is a constant matrix and $`I_{i,j}`$ is a dentity matrix. Note that the diagonal elements are $`A_{i,i}=0`$.

### Iteration process ###

The iteration process with the relaxation factor $`\omega_{opt}`$ is written as

```math
\mathbf{x_2}^{(k+1)}=\omega_{opt}\left(\mathbf{b}- \mathbf{D_1} \cdot \mathbf{A} \cdot \mathbf{D_2} \right) + \left(1 - \omega_{opt}\right)\cdot \mathbf{x_2}^{(k)},
```

where the relaxation parameter is the function of the maximal dimensionless bubble radius
```math
\omega_{opt}^{approx}=b+a\cdot\exp(c\cdot \max_i(x_i)).
```
The values of fitted coefficients $`a`$, $`b`$, and $`c`$ depend on the bubble number.

| N | $`a`$ | $`b`$ | $`c`$ |
|---:|:----:|:----:|:---:|
| 32 | 0.5 | 0.5 | -0.04 |
| 64 | 0.6 | 0.6 | -0.06 |
|128 | 0.7 | 0.3 | -0.08 |



## Project Structure ##
```text
PI-WJI/
├─ data_repository/         # Input data (raw or small sample configs)
│  └─ cluster_data/         # Cluster configurations (inputs)
│  ├─ metrics/              # Solver metrics (error, iterations, etc.)
│  ├─ runtimes/             # Total runtimes
│  ├─ time_series/          # Time-series outputs
│  └─ cluster_data/         # Derived cluster states per run
├─ figure_repository/       # Plots saved from notebooks/runs
├─ html/                    # Plotly-rendered HTML reports
├─ notebooks/
│  ├─ paper_figs.ipynb                  # Paper figures
│  ├─ sim_results_N_dependence.ipynb    # (N-dependence)
│  ├─ sim_results_PA_dependence.ipynb   # (P_A-dependence)
│  └─ tutorial.ipynb                    # Colab/Jupyter tutorial
├─ src/
│  ├─ bubble_models/        # ODE models with different linear solvers
│  ├─ utils/                # Utility functions
│  ├─ bubble_size_and_spatial_distribution_generator.py
│  └─ solver_bubble_cluster.py
├─ LICENSE
└─ README.md
```

## Quick Start ##

Two scripts are provided in `src/`:

- `bubble_size_and_spatial_distribution_generator.py` — generates a cluster configuration

- `solver_bubble_cluster.py` — runs a simulation; if no cluster exists, it can generate one

Run them from the repository root.

### A) Run a simulation (auto-generate cluster if needed)

`solver_bubble_cluster.py` accepts CLI arguments to configure the cluster and the solver. If a cluster configuration exists in `data_repository/cluster_data`, it will be reused; otherwise, a new one is generated.

```
python src/solve_bubble_cluster.py  
    --cluster.number-of-bubbles 32
    --cluster.seed 42
    --solver.bubble-model jacobi_fitted
    --solver.jacobi_params 0.5 0.5, -0.04
    --solver.measure runtime
    --solver.repeat-measure 10 
    --general.P1 1.2
```

This command creates (or loads) a 32-bubble cluster with seed 42 and solves it using PI-WJI with $`\omega_{opt}`$ with coefficient $`a=0.5`$, $`b=0.5`$, $`c=-0.04`$. 
It measures total runtime (seconds) and repeats the measurement 10 times. Excitation amplitude is $`P_1=1.2\, \mathrm{bar}`$.

For the full list of options:

```
python src/solve_bubble_cluster.py --help
```

### B) Generate a cluster first
If you prefer to create the cluster configuration first and reuse it across simulations, run `bubble_size_and_spatial_ditribution_generator.py` first. For example:

```
python src/bubble_size_and_spatial_ditribution_generator.py
    --cluster.number-of-bubble 32
    --cluster.seed 42
```

For the full list of generator flags type
```
python src/bubble_size_and_spatial_ditribution_generator.py --help
```

### C) Shell example (parameter study)
The recommended method for running simulations is to use shell scripts. For example, the shell script below searches for optimal relaxation parameter using a brute-force technique (`jacobi_bruteforce`) in a cluster configured for `N=32` bubbles and `SEED=21`.
`P2=25.0`means the excitation freuqency was $`25.0\, \mathrm{kHz}`$.

```
#!/usr/bin/env bash

cd "$(dirname "$0")"

PYTHON=python
SCRIPT=src/solve_bubble_cluster.py

# General Parameters
P2=25.0
N=32
SEED=21


for P1 in 0.6 0.8 1.0 1.2; do
echo ">>> $P1"
  $PYTHON "$SCRIPT" --cluster.number-of-bubbles $N --cluster.seed $SEED  \
                    --solver.measure linalg \
                    --solver.bubble_model jacobi_bruteforce \
                    --general.P1 $P1 --general.P2 $P2

done


read -p "Press enter to exit..."
```

### Colab Tutorial
A simple Colab notebook is provided to reproduce key results.

<a target="_blank" href="https://colab.research.google.com/https://github.com/kalmi901/PI-WJI/blob/main/notebooks/tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>


## Citation
If you use this code, please cite the corresponding paper

```
@article{PIWJI,
  title   = {Physics-Informed Weighted Jacobi for Coupled Bubble Dynamics},
  author  = {Hegedűs, F. and Kozák, Á. and Klapcsik, K.},
  journal = {TODO},
  year    = {TODO},
}
```
