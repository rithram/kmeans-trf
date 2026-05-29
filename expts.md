# Experimental details

The following is assuming that we have a directory named `results` already created at the top of the codebase and that we have a conda environment named `kmt`. Please modify the following accordingly for a different environment name.

#### Setting up the environment

This is also present in the README.

Assuming CUDA is properly setup on the machine, we will be using python version 3.13

```
> cd kmeans-trf
> conda create -n kmt
> conda activate kmt
> conda install python=3.13 pip>25.0
> conda install cudatoolkit -c anaconda  # <== OPTIONAL: if we have access to a GPU
> pip install -r requirements.txt
> conda deactivate
```

#### Creating the directory for the training runs and the results

```
> mkdir runs
> mkdir results
```


## Training runs

Training runs for Figure 9

```
for TRIAL in {1..10}; do
  for D in 4 32; do
    for LI in 1.0 4.0 7.0 10.0; do
      ENVNAME="kmt"; N=512; K=6; LR=0.01; T=10000; bash expt-run.sh ${N} ${D} ${K} ${LR} ${LI} ${T} ${TRIAL} ${ENVNAME}
    done
  done
done
```

Plotting for Figure 9a


```
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=6_d=4_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=*_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O results -S 'l' -V "e:L" --numeric -T "Loss temp" -M 1.5 --logy -C "fig9a" \
  --quantile 25 --drop_step_zero --inv_hp_val  --nlegendcols 1 --xticks_steps 5000 --alt_plot
```

Plotting for Figure 9b


```
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=6_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=*_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O results -S 'l' -V "e:L" --numeric -T "Loss temp" -M 1.5 --logy -C "fig9b" \
  --quantile 25 --drop_step_zero --inv_hp_val  --nlegendcols 1 --xticks_steps 5000 --alt_plot
```

Training runs for Figure 10a

```
for TRIAL in {1..10}; do
  for K in 10 16 25; do
    ENVNAME="kmt"; N=512; D=32; LI=10.0; LR=0.01; T=10000; bash expt-run.sh ${N} ${D} ${K} ${LR} ${LI} ${T} ${TRIAL} ${ENVNAME}
  done
done
```

Plotting for Figure 10a

```
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=*_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O results -S 'k' -V "e:L" --numeric -T "nclusters" -M 1.5 --logy -C "fig10a" \
  --quantile 25 --drop_step_zero --hp_val_int --nlegendcols 1 --xticks_steps 5000 --alt_plot
```


Training runs for Figure 10b

```
for TRIAL in {1..10}; do
  for D in 8 16; do
    ENVNAME="kmt"; N=512; LI=10.0; K=6; LR=0.01; T=10000; bash expt-run.sh ${N} ${D} ${K} ${LR} ${LI} ${T} ${TRIAL} ${ENVNAME}
  done
done
```

Plotting for Figure 10a

```
python plot_train_id_gen.py -I runs \
  -R "n=512_N=512_k=6_d=*_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O results -S 'd' -V "e:L" --numeric -T "ndims" -M 1.5 --logy -C "fig10b" \
  --quantile 25 --drop_step_zero --hp_val_int --nlegendcols 1 --xticks_steps 5000 --alt_plot
```

## Main paper plots


### Figure 4

For figure 4a, run the following (assuming the training run results are in the `runs` directory)

```
python plot_eval_ood_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=onehot_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O results -V "e:L" -M 2.7 -P "fig4a" --niters 20 --opmode mstep --nlegendcols 1 --seed 'E' \
  --init_scheme random --quantile 0 --skip_rembed_false --alt_plot
```

For figure 4b, run the following (assuming the training run results are in the `runs` directory)

```
python plot_eval_ood_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=none_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O results -V "e:L" -M 2.7 -P "fig4b" --niters 20 --opmode mstep --nlegendcols 1 --seed 'E' \
  --init_scheme random --quantile 0 --skip_rembed_false --alt_plot
```

For figure 4c, run the following (assuming the training run results are in the `runs` directory)

```
python plot_eval_ood_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=onehot_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O results -V "e:L" -M 2.7 -P "fig4c" --niters 20 --opmode mstep --nlegendcols 1 --seed 'E' \
  --init_scheme kmeams++ --quantile 0 --skip_rembed_false --alt_plot
```

For figure 4d, run the following (assuming the training run results are in the `runs` directory)

```
python plot_eval_ood_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=onehot_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O results -V "e:L" -M 2.7 -P "fig4d" --niters 20 --opmode lengen --nlegendcols 1 --seed 'E' \
  --init_scheme random --quantile 0 --skip_rembed_false --alt_plot
```

### Figure 5

### Figure 6

### Table 1 and Figure 11

The following command creates Figure 11 and also outputs a table corresponding to Table 1:
```
plot_eval_ood_gen.py -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=onehot_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O results -V "e:L" -M 2.7 -P "tab1" --niters 20 --opmode varfam \
  --nlegendcols 1 --seed 'E' --init_scheme random --quantile 0 --skip_rembed_false --alt_plot
```

### Table 2

### Table 4

### Table 5

