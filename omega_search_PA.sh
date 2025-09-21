#!/usr/bin/env bash

cd "$(dirname "$0")"

PYTHON=python   # vagy csak `python` ha úgy van PATH-ban
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