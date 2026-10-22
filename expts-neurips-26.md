# Experimental Replication Guide (NeurIPS 2026 Submission)

This document is specific to the NeurIPS 2026 submission. It provides the commands needed to
replicate the experiments from that paper. For the ICML 2026 submission, see
[`expts-icml-26.md`](expts-icml-26.md).

All commands assume you are in the repository root (`kmeans-trf`) with the conda environment set
up as described in the [README](README.md). Both training and plotting need that environment (named
`kmt` in the README): the training scripts activate `ENVNAME` themselves when run with `--run`, but
for the plotting, probing and profiling commands you need to `conda activate kmt` first.

## Setup

Create the required directories at the top of the codebase:

```bash
mkdir -p runs figures ofigures infer lsurfs ddfigs
```

- `runs`: training output (checkpoints `*.pt`, statistics `*.csv`, and training plots)
- `figures`: training-curve plots (`plot_train_id_gen.py`)
- `ofigures`: probing plots (`plot_eval_ood_gen.py`)
- `lsurfs`: loss surfaces (`ll_gen.py`, `ll_plot.py`)
- `ddfigs`: data-distance heat maps (`ddhm.py`)
- `infer`: profiling data (`profile_kmeans_transformer.py`, `run_heatmap_sweep.py`)

## Training runs

The training scripts are `expt-run.sh` (normal data) and `expt-run-dist.sh` (other data
distributions). Both write to `runs/` and take the following positional arguments:

- `N`: number of points
- `D`: dimensionality
- `K`: number of clusters
- `LR`: learning rate
- `LI`: loss inverse temperature
- `T`: training steps
- `TRIAL`: trial number (1-10), which selects the corresponding line of the
  [`random_seeds`](random_seeds) file
- `DIST` (only `expt-run-dist.sh`, argument 8): training data distribution
- `ENVNAME`: conda environment name (argument 8 for `expt-run.sh`, argument 9 for
  `expt-run-dist.sh`)

The NeurIPS experiments train every combination of loss activation and scratch encoding:

```bash
--run --loss-acts "softmax sparsemax" --encs "onehot binary none"
```

That is 6 models per setting (2 loss activations x 3 encodings). The commands below use these
variables:

```bash
ENVNAME="kmt"; ACTS="softmax sparsemax"; ENCS="onehot binary none"
```

**Verifying a command before running it:** without `--run`, the scripts only print the exact
`python train.py ...` commands they would execute and exit. For example, this prints 6 commands
and trains nothing:

```bash
bash expt-run.sh --loss-acts "${ACTS}" --encs "${ENCS}" 512 32 6 0.01 10.0 10000 1 ${ENVNAME}
```

Add `--run` to train. Run `bash expt-run.sh --help` or `bash expt-run-dist.sh --help` for all
options.

**Re-running and resuming:** if a configuration is already fully trained (its `_last.pt` and
`.csv` exist in `runs/` and the `.csv` reaches the requested number of steps), `train.py` prints
"Config already trained" and exits without error, so the loops below can be re-run after an
interruption and will skip the finished models. Some configurations appear in more than one loop
below; they are trained once and skipped afterwards. If a run was interrupted midway, `train.py`
stops with an error listing its leftover files: delete them (or use a new `--odir`) and re-run.

Each loop below trains 10 trials x 6 models (2 loss activations x 3 encodings) per setting. Some
figures draw on several loops:

| Loop | Used by |
|---|---|
| Figures 2a, 3 and 10 | Fig 2a, Fig 3, Fig 10, Fig 2b (its `d = 4`, `LI = 4.0` models), and the `k = 6` normal models in Fig 16a/c |
| Figure 11 | Fig 11 |
| Figure 13 | Fig 13 |
| Figure 14 | Fig 14 and Fig 2b (its `d = 16`, `LI = 4.0` models) |
| Figure 15 | Fig 15, and the `normal`, `k = 10`, `d = 32` models used by Figs 4, 5, 6, 7, 8, 9, 12, 16, 18 and 19 |
| Other distribution families | The `cauchy`, `laplace`, `gumbel` and `lognormal` models in Figs 4, 5, 8, 9, 16, 18 and 19 |

Models for Figures 2a, 3 and 10

```bash
for TRIAL in {1..10}; do
  for D in 4 32; do
    for val in 1.0 4.0 7.0 10.0; do
      N=512; K=6; LR=0.01; LI=${val}; T=10000
      bash expt-run.sh --run --loss-acts "${ACTS}" --encs "${ENCS}" ${N} ${D} ${K} ${LR} ${LI} ${T} ${TRIAL} ${ENVNAME}
    done
  done
done
```

Models for Figure 11

```bash
for TRIAL in {1..10}; do
  for D in 4 32; do
    for val in 1.0 0.1 0.01 0.001; do
      N=512; K=6; LR=${val}; LI=10.0; T=5000
      bash expt-run.sh --run --loss-acts "${ACTS}" --encs "${ENCS}" ${N} ${D} ${K} ${LR} ${LI} ${T} ${TRIAL} ${ENVNAME}
    done
  done
done
```

Models for Figure 13

```bash
for TRIAL in {1..10}; do
  for D in 4 32; do
    for val in 128 256 512 1024; do
      N=${val}; K=6; LR=0.01; LI=10.0; T=10000
      bash expt-run.sh --run --loss-acts "${ACTS}" --encs "${ENCS}" ${N} ${D} ${K} ${LR} ${LI} ${T} ${TRIAL} ${ENVNAME}
    done
  done
done
```

Models for Figure 14 (and Figure 2b)

```bash
for TRIAL in {1..10}; do
  for LI in 4.0 10.0; do
    for val in 8 16 32; do
      N=512; D=${val}; K=6; LR=0.01; T=10000
      bash expt-run.sh --run --loss-acts "${ACTS}" --encs "${ENCS}" ${N} ${D} ${K} ${LR} ${LI} ${T} ${TRIAL} ${ENVNAME}
    done
  done
done
```

Models for Figure 15 (and the `normal` models for Figures 4-9, 12, 16, 18 and 19)

```bash
for TRIAL in {1..10}; do
  for D in 4 32; do
    for LI in 1.0 10.0; do
      for val in 10 16 25; do
        N=512; K=${val}; LR=0.01; T=10000
        bash expt-run.sh --run --loss-acts "${ACTS}" --encs "${ENCS}" ${N} ${D} ${K} ${LR} ${LI} ${T} ${TRIAL} ${ENVNAME}
      done
    done
  done
done
```

Models for the other distribution families (Figures 4, 5, 8, 9, 16, 18 and 19)

```bash
for TRIAL in {1..10}; do
  for DIST in cauchy laplace gumbel lognormal; do
    for D in 32; do
      for LI in 1.0 10.0; do
        for K in 6 10; do
          N=512; LR=0.01; T=10000
          bash expt-run-dist.sh --run --loss-acts "${ACTS}" --encs "${ENCS}" ${N} ${D} ${K} ${LR} ${LI} ${T} ${TRIAL} ${DIST} ${ENVNAME}
        done
      done
    done
  done
done
```

## Main paper plots

All plotting commands read the trained models and statistics from `runs/`.

### Figure 1

We will sweep various $(d, k)$ values and profile the forward pass and backward pass times, and the GPU memory overhead. The results and the figure will be saved in the `infer` directory.

```bash
python run_heatmap_sweep.py
```

This `run_heatmap_sweep.py` script internally calls `profile_kmeans_transformer.py` with different values of $(d,k)$.

**GPU:** Figure 1 and Table 1 in the paper were generated on a GPU, so run these profiling commands on a GPU. `run_heatmap_sweep.py` uses `--device auto`, which picks CUDA when it is available. On a CPU the timings still run, but the GPU memory columns are `NaN`.

**Possible file collisions:** Figure 1 and Table 1 both write `infer/d{d}_k{k}.csv`, and they share two configurations, `d = 32, k = 25` and `d = 32, k = 100`, with identical profiling settings. `run_heatmap_sweep.py` skips a configuration whose CSV already exists, so whichever of the two is run second may reuse or overwrite the other's file. This is benign when both are run on the same device (the GPU, for the paper's numbers), since they measure the same thing. If you switch devices, delete the CSVs in `infer/` first.

### Table 1

The data for the table is generated with the following command:

```bash
for k in 6 25 100 400 1000; do
  python profile_kmeans_transformer.py --d 32 --k ${k} \
    --task-batch-size 32 --points-per-batch 1024 --iters 30 --warmup-iters 10 \
    --seed 548977 --device cuda --csv infer/d32_k${k}.csv
done
```

Table 1 in the paper was generated on a GPU (see the GPU and file-collision notes under Figure 1). `--device cpu` also works, but the memory columns are then `NaN`.


### Figure 2

Figure 2a

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=6_d=4_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=1*0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'l' -V "e:L" --numeric -T "Reg pen $\tau$" -M 2.05 --logy -C "N512d4k6T10k_cc" \
  --quantile 25 --drop_step_zero --inv_hp_val  --nlegendcols 2 --xticks_steps 2500
```

Figure 2b

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=6_d=[14][6_]*D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=4.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'd' -V "e:L" --numeric -T "dims d" -M 2.05 --logy -C "N512k6T10kLI4_cc" \
  --quantile 25 --drop_step_zero --hp_val_int --nlegendcols 2 --xticks_steps 2500
```

The results will be saved in the `figures` directory.

### Figure 3

The loss surfaces (data and plots) will be saved in the `lsurfs` directory. Figure 3 uses a single checkpoint per configuration: trial 1 (`E=153476998`, the first line of [`random_seeds`](random_seeds)), not an aggregate over the 10 trials. The loss surfaces are generated from the trained models as follows:

```bash
for ENC in onehot binary; do
  for LA in sparsemax; do
    for NDIMS in 4 32; do
      python ll_gen.py \
        --checkpoint runs/n=512_N=512_k=6_d=${NDIMS}_D=normal_s=0.1_em=True_es=True_sdb=True_e=${ENC}_q=1_A=softmax_a=1.0_p=0.01_L=${LA}_g=False_l=10.0_E=153476998_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt \
        --pseed 1111 --nsteps 200 --wbsz 1000  --output lsurfs/
    done
  done
done
```

We plot the loss surfaces in Figures 3a-d as follows, again saving in the `lsurfs` directory. The panels are (a) `d = 4` with `onehot` (OH), (b) `d = 4` with `binary` (BN), (c) `d = 32` with `onehot`, and (d) `d = 32` with `binary`; each is a separate output file, named after its input `--lldata` file:

```bash
for ENC in onehot binary; do
  for LA in sparsemax; do
    for NDIMS in 4 32; do
      python ll_plot.py \
        --lldata lsurfs/n=512_N=512_k=6_d=${NDIMS}_D=normal_s=0.1_em=True_es=True_sdb=True_e=${ENC}_q=1_A=softmax_a=1.0_p=0.01_L=${LA}_g=False_l=10.0_E=153476998_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10__aub:1.0__nsteps:200__pseed:1111__lsurf.csv \
        --output lsurfs --nlevels 20 --fig_type contour --metric ub
    done
  done
done
```

### Figure 4

Figure 4a

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=*_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=1.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'D' -V "e:L" -T "D" -M 2.5 --logy -C "N512d32k10T10kLI1_cc" \
  --quantile 25 --drop_step_zero --nlegendcols 1 --sharey --xticks_steps 5000
```

Figure 4b

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=*_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'D' -V "e:L" -T "D" -M 2.5 --logy -C "N512d32k10T10kLI10_cc" \
  --quantile 25 --drop_step_zero --nlegendcols 1 --sharey --xticks_steps 5000
```

The results will be saved in the `figures` directory.

### Figure 5

Figure 5 also uses a single checkpoint per configuration (trial 1, `E=153476998`). We generate the `cauchy` and `normal` loss landscape data in the `lsurfs` directory as follows:

```bash
for ENC in onehot; do
  for LA in sparsemax; do
    for NDIMS in 32; do
      for DIST in cauchy normal; do
        python ll_gen.py \
          --checkpoint runs/n=512_N=512_k=10_d=${NDIMS}_D=${DIST}_s=0.1_em=True_es=True_sdb=True_e=${ENC}_q=1_A=softmax_a=1.0_p=0.01_L=${LA}_g=False_l=10.0_E=153476998_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt \
          --pseed 1111 --nsteps 200 --wbsz 1000  --output lsurfs/
      done
    done
  done
done
```

Then we plot the loss gap landscape in the `lsurfs` directory as follows:

```bash
for ENC in onehot; do
  for LA in sparsemax; do
    for NDIMS in 32; do
      for DIST in normal cauchy; do
        python ll_plot.py \
          --lldata lsurfs/n=512_N=512_k=10_d=${NDIMS}_D=${DIST}_s=0.1_em=True_es=True_sdb=True_e=${ENC}_q=1_A=softmax_a=1.0_p=0.01_L=${LA}_g=False_l=10.0_E=153476998_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10__aub:1.0__nsteps:200__pseed:1111__lsurf.csv \
          --output lsurfs --fig_type heatmap --metric diff
      done
    done
  done
done
```


## Probing

This assumes the presence of the directory `ofigures`.

### Figure 6

```bash
python plot_eval_ood_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O ofigures -V "e:L" -M 2.4 -P "n512d32k10l10T10k" --niters 20 \
  --lossact sparsemax --nlegendcols 1 --seed 'E' --opmode vardist
```

The results will be saved in the `ofigures` directory.

### Figure 7

```bash
python plot_eval_ood_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O ofigures -V "e:L" -M 2.7 -P "n512d32k10l10T10k" --niters 20 \
  --lossact sparsemax --opmode mstep --nlegendcols 2 --seed 'E'
```

For `mstep`, `--lossact` only affects the output file name: the figure shows both the `softmax` (NE) and `sparsemax` (L2) models, as in the paper. The results will be saved in the `ofigures` directory.

### Figure 8 (Figure 18)

This is the `crossfam` opmode: models trained on one distribution family are evaluated on the others.

```bash
python plot_eval_ood_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=*_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O ofigures -V "e:L" -M 2.4 -P "n512d32k10l10T10k" --niters 20 \
  --lossact sparsemax --nlegendcols 1 --seed 'E' --opmode crossfam
```

The results will be saved in the `ofigures` directory.

### Figure 9 and 19

`--ntasks 4` gives Figure 9 and `--ntasks 16` gives Figure 19. These use a single checkpoint per distribution family (trial 1, `E=153476998`), and only the validation tasks stored in it, not the model weights. The results will be saved in the `ddfigs` directory. The order of the family blocks in the heat maps follows the order in which the checkpoint files are listed (`ddhm.py` does not sort them), so it may differ from the `normal, cauchy, gumbel, laplace, lognormal` order shown in the paper.

```bash
for NTASKS in 4 16; do
  python ddhm.py \
    --cprg "runs/n=512_N=512_k=10_d=32_D=*_s=0.1_em=True_es=True_sdb=True_e=onehot_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=153476998_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
    --output ddfigs --ntasks ${NTASKS}
done
```

## Ablations

### Figure 10

Figures 10a-b

```bash
for NDIMS in 4 32; do
  python plot_train_id_gen.py -I runs \
    -R "n=512_N=512_k=6_d=${NDIMS}_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=*_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
    -O figures -S 'l' -V "e:L" --numeric -T "Loss pen" -M 2.7 --logy -C "N512d${NDIMS}k6T10k" \
    --quantile 25 --drop_step_zero --inv_hp_val
done
```

The results will be saved in the `figures` directory.

### Figure 11

Figure 11a

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=6_d=4_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=*_C=0.5_P=5.0_T=5000_t=50_B=10.csv" \
  -O figures -S 'r' -V "e:L" --numeric -T "LR $\eta$" -M 2.7 --logy -C "N512d4k6T5k" \
  --quantile 25 --drop_step_zero --xticks_steps 2000
```

Figure 11b

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=6_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=*_C=0.5_P=5.0_T=5000_t=50_B=10.csv" \
  -O figures -S 'r' -V "e:L" --numeric -T "LR $\eta$" -M 2.7 --logy -C "N512d32k6T5k" \
  --quantile 25 --drop_step_zero --xticks_steps 2000
```

The results will be saved in the `figures` directory.

### Figure 12

```bash
python plot_eval_ood_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O ofigures -V "e:L" -M 2.4 -P "n512d32k10l10T10k" --niters 20 \
  --lossact sparsemax --nlegendcols 1 --seed 'E' --opmode lengen
```

The results will be saved in the `ofigures` directory.

### Figure 13

Figure 13a

```bash
python plot_train_id_gen.py -I runs \
  -R "n=*_N=*_k=6_d=4_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'n' -V "e:L" --numeric -T "nsamples" -M 2.7 --logy -C "d4k6T10k" \
  --quantile 25 --drop_step_zero --hp_val_int
```

Figure 13b

```bash
python plot_train_id_gen.py -I runs \
  -R "n=*_N=*_k=6_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'n' -V "e:L" --numeric -T "nsamples" -M 2.7 --logy -C "d32k6T10k" \
  --quantile 25 --drop_step_zero --hp_val_int
```

The results will be saved in the `figures` directory.

### Figure 14

Figure 14a ($\tau = 0.25$)

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=6_d=*_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=4.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'd' -V "e:L" --numeric -T "ndims" -M 2.7 --logy -C "N512k6T10kLI4" \
  --quantile 25 --drop_step_zero --hp_val_int
```

Figure 14b ($\tau = 0.1$)

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=6_d=*_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'd' -V "e:L" --numeric -T "ndims" -M 2.7 --logy -C "N512k6T10kLI10" \
  --quantile 25 --drop_step_zero --hp_val_int
```

The results will be saved in the `figures` directory.

### Figure 15

Figure 15a

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=*_d=4_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=1.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'k' -V "e:L" --numeric -T "nclusters" -M 2.7 --logy -C "N512d4T10kLI1" \
  --quantile 25 --drop_step_zero --hp_val_int
```

Figure 15b

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=*_d=4_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'k' -V "e:L" --numeric -T "nclusters" -M 2.7 --logy -C "N512d4T10kLI10" \
  --quantile 25 --drop_step_zero --hp_val_int
```

Figure 15c

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=*_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=1.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'k' -V "e:L" --numeric -T "nclusters" -M 2.7 --logy -C "N512d32T10kLI1" \
  --quantile 25 --drop_step_zero --hp_val_int
```

Figure 15d

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=*_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'k' -V "e:L" --numeric -T "nclusters" -M 2.7 --logy -C "N512d32T10kLI10" \
  --quantile 25 --drop_step_zero --hp_val_int
```

The results will be saved in the `figures` directory.

### Figure 16

Figure 16a

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=6_d=32_D=*_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=1.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'D' -V "e:L" -T "D" -M 2.5 --logy -C "N512d32k6T10kLI1" \
  --quantile 25 --drop_step_zero --nlegendcols 1 --sharey --xticks_steps 5000
```

Figure 16b

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=*_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=1.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'D' -V "e:L" -T "D" -M 2.5 --logy -C "N512d32k10T10kLI1" \
  --quantile 25 --drop_step_zero --nlegendcols 1 --sharey --xticks_steps 5000
```

Figure 16c

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=6_d=32_D=*_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'D' -V "e:L" -T "D" -M 2.5 --logy -C "N512d32k6T10kLI10" \
  --quantile 25 --drop_step_zero --nlegendcols 1 --sharey --xticks_steps 5000
```

Figure 16d

```bash
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=*_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=*_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O figures -S 'D' -V "e:L" -T "D" -M 2.5 --logy -C "N512d32k10T10kLI10" \
  --quantile 25 --drop_step_zero --nlegendcols 1 --sharey --xticks_steps 5000
```

The results will be saved in the `figures` directory.

### Figure 17

🚧 **Under construction.** The commands for Figure 17 (training and validation loss against actual elapsed training time, for the `OH` and `BN` embeddings) are not included yet. We are working on it and will add this as soon as possible.
