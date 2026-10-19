#!/usr/bin/env python

import argparse
import os

import matplotlib.colors as mcols
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_tensor_col(values: np.ndarray) -> np.ndarray:
    # columns written by ll_gen.py may hold strings like "tensor(1.2345)"
    return np.array([float(str(v).replace("tensor(", "").replace(")", "")) for v in values])


def metric_obj(data: pd.DataFrame, metric: str) -> np.ndarray:
    if metric in ["obj", "ub"]:
        m = "train-loss" if metric == "ub" else "kmeans-obj"
        return parse_tensor_col(data[m].values)
    objs = parse_tensor_col(data["kmeans-obj"].values)
    ubs = parse_tensor_col(data["train-loss"].values)
    diff = ubs - objs
    if metric == "diff":
        return diff
    rdiff = diff / objs
    return rdiff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lldata",
        help="Loss surface data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Directory for output",
        type=str,
        default="",
    )
    parser.add_argument(
        "--nlevels",
        help="Number of loss levels",
        type=int,
        default=20,
    )
    metrics = ["ub", "obj", "diff", "rdiff"]
    parser.add_argument(
        "--metric",
        help="Metric to plot",
        choices=metrics,
        default=metrics[0],
    )
    ftype = ["contour", "heatmap"]
    parser.add_argument(
        "--fig_type",
        help="What figure to plot",
        choices=ftype,
        required=True,
    )
    parser.add_argument(
        "--log_metric",
        help="Apply log on the metric before plotting",
        action="store_true",
    )

    args = parser.parse_args()
    assert os.path.exists(args.lldata)
    assert os.path.exists(args.output)
    assert args.nlevels > 3
    fname_prefix = args.lldata.split("/")[-1].replace(".csv", "")
    fname = fname_prefix + f"_{'log_' if args.log_metric else ''}{args.metric}_{args.fig_type}.pdf"
    fpath = os.path.join(args.output, fname)
    print(f"Saving figures in \n{fpath}")

    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(4 * ncols, nrows * 4), squeeze=True)

    # load up data
    lldata = pd.read_csv(args.lldata)
    print(lldata.shape)
    print(list(lldata))
    xdata = np.sort(lldata["xval"].unique())
    ydata = np.sort(lldata["yval"].unique())
    print(xdata.shape, ydata.shape)
    nvals = len(xdata)
    assert nvals == len(ydata)
    lgrid = np.empty([nvals, nvals])
    xmin = xdata.min()
    ymin = ydata.min()
    xgap = xdata[1] - xdata[0]
    ygap = ydata[1] - ydata[0]
    xidx = ((lldata["xval"].values - xmin) / xgap).astype(int)
    print(xidx.min(), xidx.max(), xidx.shape)
    yidx = ((lldata["yval"].values - ymin) / ygap).astype(int)
    print(yidx.min(), yidx.max(), yidx.shape)
    objs = metric_obj(lldata, args.metric)
    if args.log_metric:
        objs = np.log(objs)
    print(type(objs))
    print(type(objs[0]))
    print(objs[0], objs[-1])
    print(objs.min(), objs.max())
    for xx, yy, vv in zip(xidx, yidx, objs, strict=True):
        lgrid[xx, yy] = vv
    if args.fig_type == "contour":
        lmin, lmax = objs.min(), objs.max()
        levels = np.exp(np.linspace(np.log(lmin), np.log(lmax), num=args.nlevels))
        print(f"Levels: {levels}")
        # Create Contour Plot
        print("Creating and plotting contours ...")
        ax.set_aspect("equal")
        CS = ax.contour(
            xdata,
            ydata,
            lgrid,
            levels=levels,
            cmap="magma",
            linewidths=0.75,
            norm=mcols.LogNorm(vmin=lmin, vmax=lmax * 2.0),
        )
        ax.clabel(CS, inline=True, fontsize=8, fmt="%1.2f")

    if args.fig_type == "heatmap":
        # Create heatmap
        print("Plotting heatmap ...")
        ax.set_aspect("equal")
        im = ax.imshow(lgrid, cmap="inferno_r")
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
        ax.tick_params(axis="both", labelbottom=False, labelleft=False)

    fig.tight_layout()
    print(f"Saving figure in {fpath} ...")
    fig.savefig(fpath)
