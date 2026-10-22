#!/usr/bin/env python

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from scipy.stats import wasserstein_distance as WD

from ctasks import DMAP

np.set_printoptions(precision=2)
pd.set_option("display.precision", 2)
print(f"Cuda available: {torch.cuda.is_available()}")
GPUE = torch.cuda.is_available()


def get_dists(Q: torch.Tensor, R: torch.Tensor, sym: bool = False) -> np.ndarray:
    vals = None
    nq, nr = Q.shape[0], R.shape[0]
    ne = (nq * (nq - 1) // 2) if sym else (nq * nr)
    dists = torch.square(torch.cdist(Q, R))
    if sym:
        assert dists.shape[0] == dists.shape[1]
        ltr = torch.tril_indices(dists.shape[0], dists.shape[1], offset=-1)
        vals = dists[ltr[0], ltr[1]]
    else:
        vals = dists.reshape(
            -1,
        )
    assert vals.shape[0] == ne, f"Expected {ne}, got {vals.shape}"
    return vals.to("cpu").numpy()


def get_dist_data(
    cfile: str, max_ntasks: int
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    dname = cfile.split("/")[-1].replace("_last.pt", "").split("_")[4].split("=")[1]
    assert dname in DMAP.keys()
    key = dname.upper()[:2]
    print(f"Using key: '{key}' for distribution '{dname}'...")
    checkpoint = torch.load(cfile)
    BX, BC = checkpoint["val_tasks"][0]
    X, C = BX[:max_ntasks], BC[:max_ntasks]
    ntasks = X.shape[0]
    assert C.shape[0] == ntasks
    names = [f"{key}{i + 1}" for i in range(ntasks)]
    xdists = [get_dists(XX, XX, sym=True) for XX in X]
    cdists = [get_dists(CC, CC, sym=True) for CC in C]
    cxdists = [get_dists(XX, CC, sym=False) for XX, CC in zip(X, C, strict=True)]
    return names, xdists, cdists, cxdists


def heatmap(
    hax: Axes,
    hgrid: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    blabel: str,
    mult: int,
    cmap: str = "YlGn",
) -> None:
    im = hax.imshow(hgrid, cmap=cmap)
    hax.figure.colorbar(im, ax=hax, shrink=0.75)
    # cbar.ax.set_ylabel(blabel, rotation=-90, va="bottom", fontsize=10)
    hax.set_title(blabel, fontsize=4 * mult)
    hax.set_xticks(
        range(len(xlabels)),
        labels=xlabels,
        rotation=90,
        ha="left",
        rotation_mode="anchor",
        fontsize=8,
    )
    hax.set_yticks(range(len(ylabels)), labels=ylabels, fontsize=8)
    hax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cprg",
        help="Regular expression for model checkpoints",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Directory for output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ntasks", help="Number of tasks per distribution to consider", type=int, default=5
    )
    args = parser.parse_args()
    assert os.path.exists(args.output)
    cpfiles = glob.glob(args.cprg)
    assert len(cpfiles) > 0
    outfile = f"dd_heatmap_{args.ntasks}.pdf"
    outfile = os.path.join(args.output, outfile)
    print(f"Saving the heatmap in {outfile}")

    # extract tasks pairwise distances
    print(f"Extracting data from {len(cpfiles)} checkpoints ...")
    N, XX, CC, XC = [], [], [], []
    for cp in cpfiles:
        n, xx, cc, xc = get_dist_data(cp, args.ntasks)
        N += n
        XX += xx
        CC += cc
        XC += xc
    print(f"Data extracted for {N}\n{np.stack(XX).shape}, {np.stack(XC).shape}")

    # compute pairwise WD + plot
    dlist = [CC, XC, XX]
    titles = ["Inter-center sq dists", "Point-center sq dists", "Inter-point sq dists"]
    nrows, ncols = 1, len(dlist)
    mult = args.ntasks
    fsize = (mult * ncols, mult * nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=fsize)
    for i in range(ncols):
        ax = axs[i]
        dist = dlist[i]
        print(titles[i])
        ndists = len(dist)
        dgrid = np.empty([ndists, ndists])
        for j in range(ndists):
            for k in range(ndists):
                dgrid[j, k] = WD(dist[j], dist[k])
        print(dgrid.min(), dgrid.max())
        heatmap(ax, dgrid, N, N, titles[i], args.ntasks)

    fig.tight_layout()
    print(f"Saving figure in {outfile} ...")
    fig.savefig(outfile)
