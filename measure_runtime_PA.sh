#!/usr/bin/env bash

cd "$(dirname "$0")"

PYTHON=python   # vagy csak `python` ha úgy van PATH-ban
SCRIPT=src/solve_bubble_cluster.py

# General Parameters
P2=25.0
N=128
SEED=21

# N alapján kiválasztjuk a jacobi paramétereket
case "$N" in
  32)  JACOBI_PARAMS=(0.5 0.5 -0.04) ;;
  64)  JACOBI_PARAMS=(0.6 0.4 -0.06) ;;
  128) JACOBI_PARAMS=(0.7 0.3 -0.08) ;;
  *)
    echo "Unknown N=$N — no fitted parametes --solver.jacobi-params." >&2
    exit 1
    ;;
esac

echo ">>> JACOBI PARAMS ${JACOBI_PARAMS[@]}"


for P1 in 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2; do
  echo ">>> BASELINE: $P1"
  echo ">>> TIME: $(date +"%H:%M:%S")"
  $PYTHON "$SCRIPT" --cluster.number-of-bubbles $N --cluster.seed $SEED  \
                    --solver.measure runtime \
                    --solver.bubble_model baseline \
                    --general.P1 $P1 --general.P2 $P2
  
  echo ">>> JACOBI FITTED: $P1"
  echo ">>> TIME: $(date +"%H:%M:%S")"
  $PYTHON "$SCRIPT" --cluster.number-of-bubbles $N --cluster.seed $SEED  \
                    --solver.measure runtime \
                    --solver.bubble_model jacobi_fitted \
                    --solver.jacobi-params "${JACOBI_PARAMS[@]}" \
                    --general.P1 $P1 --general.P2 $P2
  
  
  
  echo ">>> BICG: $P1"
  echo ">>> TIME: $(date +"%H:%M:%S")"
  $PYTHON "$SCRIPT" --cluster.number-of-bubbles $N --cluster.seed $SEED  \
                    --solver.measure runtime \
                    --solver.bubble_model bicg \
                    --general.P1 $P1 --general.P2 $P2

  echo ">>> BICGSTAB: $P1"
  echo ">>> TIME: $(date +"%H:%M:%S")"
  $PYTHON "$SCRIPT" --cluster.number-of-bubbles $N --cluster.seed $SEED  \
                    --solver.measure runtime \
                    --solver.bubble_model bicgstab \
                    --general.P1 $P1 --general.P2 $P2

done


for P1 in  0.6 0.8 1.0 1.2; do

  echo ">>> JACOBI BASELINE: $P1"
  echo ">>> TIME: $(date +"%H:%M:%S")"
  $PYTHON "$SCRIPT" --cluster.number-of-bubbles $N --cluster.seed $SEED  \
                    --solver.measure runtime \
                    --solver.bubble_model jacobi_baseline \
                    --general.P1 $P1 --general.P2 $P2

done


read -p "Press enter to exit..."