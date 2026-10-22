#!/bin/bash

usage() {
  cat << EOF
Usage: bash expt-run.sh [options] N D K LR LI T TRIAL [ENVNAME]

Trains one k-means transformer per (loss activation, scratch encoding) pair
using the hyperparameters below. Without --run the train.py commands are only
printed (dry run).

Positional arguments:
  N        number of data points (nlb = nub = N)
  D        number of dimensions
  K        number of clusters
  LR       initial learning rate
  LI       loss inverse temperature
  T        number of training steps
  TRIAL    repetition index 1..10; selects line TRIAL of random_seeds
  ENVNAME  conda environment to activate (required with --run)

Options:
  --run                   execute the commands (default: only print them)
  --loss-acts "<list>"    space-separated loss activations (default: "softmax")
  --encs "<list>"         space-separated scratch encodings (default: "onehot none")
  -h, --help              show this message

Examples:
  # ICML 2026 (print only, then run)
  bash expt-run.sh 512 32 10 0.01 10 10000 1 trch
  bash expt-run.sh --run 512 32 10 0.01 10 10000 1 trch

  # NeurIPS 2026 sweep (softmax/sparsemax x onehot/binary/none)
  bash expt-run.sh --run --loss-acts "softmax sparsemax" \\
    --encs "onehot binary none" 512 32 10 0.01 10 10000 1 trch
EOF
}

RUN=0
LOSSACTS="softmax"
ENCS="onehot none"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=1; shift ;;
    --loss-acts) LOSSACTS="$2"; shift 2 ;;
    --encs) ENCS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ ${#ARGS[@]} -lt 7 ]]; then
  usage >&2
  exit 1
fi

NSAMPLES=${ARGS[0]}
NDIMS=${ARGS[1]}
NCLUSTERS=${ARGS[2]}
LRATE=${ARGS[3]}
LITEMP=${ARGS[4]}
NSTEPS=${ARGS[5]}
TRIAL=${ARGS[6]}
ENVNAME=${ARGS[7]:-}

if [[ $RUN -eq 1 && -z "$ENVNAME" ]]; then
  echo "ENVNAME (8th positional argument) is required with --run" >&2
  exit 1
fi

DISTS="normal"
SCALE=0.1
DQKV=1
ATTACT="softmax"
ATTITEMP=1.0
BSZ=32
STEPS_EVAL=50
VAL_NBATCHES=10
LRDECAY=0.5
PATIENCE=5
ODIR="runs"

NAME="N${NSAMPLES}-D${NDIMS}-K${NCLUSTERS}-LR${LRATE}-LI${LITEMP}-T${NSTEPS}-${TRIAL}"
echo $NAME

if [[ $RUN -eq 1 ]]; then
  echo "Activating conda environment ${ENVNAME}..."
  eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
  conda activate ${ENVNAME}
fi

SEED=$(sed "${TRIAL}q;d" random_seeds)
echo "RUNNING REP ${TRIAL}/10 WITH SEED ${SEED} ..."

for a in ${LOSSACTS}; do
  for e in ${ENCS}; do
    echo "--------------------------------------------------"
    echo " Loss activation ${a} with encoding ${e}"
    echo "--------------------------------------------------"
    cmd="python train.py \
      --nlb ${NSAMPLES} --nub ${NSAMPLES} --nclusters ${NCLUSTERS} --ndims ${NDIMS} \
      --train_dists ${DISTS} --scale ${SCALE} --em --es --sdb \
      --scratch ${e}  --dqkv ${DQKV} \
      --attn_act ${ATTACT} --attn_itemp ${ATTITEMP} \
      --loss_act ${a} --loss_itemp ${LITEMP} \
      --seed ${SEED} --bsz ${BSZ} --init_lr ${LRATE} \
      --lr_decay ${LRDECAY} --patience ${PATIENCE} \
      --nsteps ${NSTEPS} --nsteps_per_eval ${STEPS_EVAL} \
      --val_nbatches ${VAL_NBATCHES} --odir ${ODIR}"
    echo $cmd
    if [[ $RUN -eq 1 ]]; then
      $cmd || exit 1
    fi
  done
done

echo "REP ${TRIAL}/10 completed"
if [[ $RUN -eq 1 ]]; then
  echo "Deactivating conda environment"
  conda deactivate
fi
