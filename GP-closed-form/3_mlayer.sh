#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT=3_mlayer_ann.py
OUTCSV=ann_results.csv

# Grid settings
NEURONS_LIST=(4 16 64 256 1024 4096)
DROPOUT_LIST=(0.0 0.1 0.2 0.3)
NLAYERS=1
ACTIV=relu

# Cases:
# (a) log=yes affine=yes
# (b) log=no  affine=no
declare -a CASES=(
  "yes yes"
  "no  no"
)

for case in "${CASES[@]}"; do
  read -r LOG AFFINE <<< "$case"

  # BooleanOptionalAction expects flags: --log / --no-log, --affine / --no-affine
  if [[ "$LOG" == "yes" ]]; then
    LOG_FLAG="--log"
  else
    LOG_FLAG="--no-log"
  fi

  if [[ "$AFFINE" == "yes" ]]; then
    AFFINE_FLAG="--affine"
  else
    AFFINE_FLAG="--no-affine"
  fi

  for do in "${DROPOUT_LIST[@]}"; do
    for n in "${NEURONS_LIST[@]}"; do
      echo "Running: axial log=$LOG affine=$AFFINE dropout=$do nlayers=$NLAYERS neurons=$n"

      $PY $SCRIPT \
        --axial \
        $LOG_FLAG \
        $AFFINE_FLAG \
        --dropout "$do" \
        --nlayers "$NLAYERS" \
        --neurons "$n" \
        --activ "$ACTIV" \
        --csv "$OUTCSV" \
        --no_plot
    done
  done
done

echo "Done. Results appended to: $OUTCSV"
