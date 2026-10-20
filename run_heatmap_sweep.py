#!/usr/bin/env python
"""
Sweep profile_kmeans_transformer.py over a grid of (d, k) values and produce
4 heatmaps of the relative improvement of binary over onehot for:
  1. forward_ms  (train mode)
  2. backward_ms (train mode)
  3. enc_delta_mb (train mode)
  4. forward_ms  (eval mode)

Relative improvement = (onehot - binary) / onehot * 100  (%)
Positive = binary is better (cheaper).
"""

import itertools
import os
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
D_VALUES = [4, 8, 16, 32, 64, 128]
K_VALUES = [10, 16, 25, 40, 64, 100]

TASK_BATCH_SIZE = 32
POINTS_PER_BATCH = 1024
ITERS = 30
WARMUP_ITERS = 10
SEED = 548977

SCRIPT = "profile_kmeans_transformer.py"
CSV_DIR = "infer"
OUT_DIR = "infer"

# Metrics to extract from the CSV (short names as produced by the script)
# (metric_short_name, mode_filter, heatmap_title, unit)
HEATMAPS = [
    ("f_ms", "train", "Forward time (train)", "ms"),
    ("b_ms", "train", "Backward time (train)", "ms"),
    ("edmem_ms", "train", "Encoding specific GPU memory usage (train)", "MB"),
    ("f_ms", "eval", "Forward time (eval)", "ms"),
]


def run_one(d: int, k: int) -> str:
    """Run the profiling script for one (d,k) pair and return the CSV path."""
    os.makedirs(CSV_DIR, exist_ok=True)
    csv_path = os.path.join(CSV_DIR, f"d{d}_k{k}.csv")
    if os.path.exists(csv_path):
        print(f"  [skip] d={d} k={k} — CSV already exists", flush=True)
        return csv_path

    cmd = [
        sys.executable,
        SCRIPT,
        "--d",
        str(d),
        "--k",
        str(k),
        "--task-batch-size",
        str(TASK_BATCH_SIZE),
        "--points-per-batch",
        str(POINTS_PER_BATCH),
        "--iters",
        str(ITERS),
        "--warmup-iters",
        str(WARMUP_ITERS),
        "--seed",
        str(SEED),
        "--device",
        "auto",
        "--csv",
        csv_path,
    ]
    print(f"  [run]  d={d} k={k} …", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FAILED")
        print(result.stderr[-2000:])
        raise RuntimeError(f"Script failed for d={d} k={k}")
    print("done", flush=True)
    return csv_path


def load_results() -> pd.DataFrame:
    """Collect all per-(d,k) CSVs into one DataFrame."""
    frames = []
    for d, k in itertools.product(D_VALUES, K_VALUES):
        csv_path = os.path.join(CSV_DIR, f"d{d}_k{k}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing CSV: {csv_path}")
        frames.append(pd.read_csv(csv_path))
    return pd.concat(frames, ignore_index=True)


def build_matrices(
    df: pd.DataFrame, metric: str, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns three (len(D_VALUES) x len(K_VALUES)) arrays:
      - rel_mat:  relative improvement (%) = (onehot - binary) / onehot * 100
      - oh_mean:  onehot mean metric value
      - oh_std:   onehot std metric value
    """
    std_col = metric.replace("_ms", "_sd")
    rel_mat = np.full((len(D_VALUES), len(K_VALUES)), np.nan)
    oh_mean = np.full((len(D_VALUES), len(K_VALUES)), np.nan)
    oh_std = np.full((len(D_VALUES), len(K_VALUES)), np.nan)
    mode_df = df[df["mode"] == mode]
    for i, d in enumerate(D_VALUES):
        for j, k in enumerate(K_VALUES):
            mask = (mode_df["d"] == d) & (mode_df["k"] == k)
            sub = mode_df.loc[mask]
            oh_m = sub.loc[sub["enc"] == "onehot", metric].values
            oh_s = sub.loc[sub["enc"] == "onehot", std_col].values
            bi_m = sub.loc[sub["enc"] == "binary", metric].values
            if len(oh_m) == 1 and len(bi_m) == 1:
                oh_val, bi_val = float(oh_m[0]), float(bi_m[0])
                oh_mean[i, j] = oh_val
                oh_std[i, j] = float(oh_s[0]) if len(oh_s) == 1 else np.nan
                if oh_val != 0.0:
                    rel_mat[i, j] = (oh_val - bi_val) / oh_val * 100.0
    return rel_mat, oh_mean, oh_std


def plot_heatmaps(df: pd.DataFrame, out_path: str = "heatmaps.png"):
    TICK_FS = 13  # axis tick labels
    LABEL_FS = 21  # axis labels
    TITLE_FS = 21  # per-panel titles
    CELL_FS = 15  # in-cell relative % annotation
    SUB_FS = 7  # in-cell absolute value annotation

    n = len(HEATMAPS)
    fig, axes = plt.subplots(1, n, figsize=(9.5 * n, 7))

    # Build all matrices and shared colour range
    matrix_triples = [build_matrices(df, metric, mode) for metric, mode, _, _unit in HEATMAPS]
    all_rel = np.concatenate([rel[~np.isnan(rel)] for rel, _, _ in matrix_triples])
    vmax = max(float(np.max(np.abs(all_rel))), 1.0)

    ims = []
    for ax_idx, (ax, (_metric, _mode, title, unit), (rel_mat, oh_mean, oh_std)) in enumerate(
        zip(axes, HEATMAPS, matrix_triples, strict=True)
    ):
        im = ax.imshow(rel_mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ims.append(im)

        ax.set_xticks(range(len(K_VALUES)))
        ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=TICK_FS)
        ax.set_yticks(range(len(D_VALUES)))
        ax.set_xlabel(r"$k$  (number of clusters)", fontsize=LABEL_FS)
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")

        # Only the leftmost panel gets y-axis tick labels and the shared y label
        if ax_idx == 0:
            ax.set_yticklabels([str(d) for d in D_VALUES], fontsize=TICK_FS)
            ax.set_ylabel(r"$d$  (data dimensionality)", fontsize=LABEL_FS)
        else:
            ax.set_yticklabels([])

        # Annotate each cell: relative % on top, onehot mean±std on second line
        for i in range(len(D_VALUES)):
            for j in range(len(K_VALUES)):
                rel_val = rel_mat[i, j]
                mean_v = oh_mean[i, j]
                std_v = oh_std[i, j]
                if np.isnan(rel_val):
                    continue
                text_color = "black" if abs(rel_val) < 0.6 * vmax else "white"
                if unit == "ms":
                    abs_str = f"({mean_v:.1f}±{std_v:.1f} ms)"
                else:
                    abs_str = f"({mean_v:.1f}±{std_v:.1f} MB)"
                ax.text(
                    j,
                    i - 0.12,
                    f"{rel_val:+.1f}%",
                    ha="center",
                    va="center",
                    fontsize=CELL_FS,
                    color=text_color,
                    fontweight="bold",
                )
                ax.text(
                    j,
                    i + 0.30,
                    abs_str,
                    ha="center",
                    va="center",
                    fontsize=SUB_FS,
                    color=text_color,
                )

    # Single shared colorbar attached to the rightmost axis
    cbar = fig.colorbar(ims[-1], ax=axes[-1], fraction=0.046, pad=0.04)
    cbar.set_label("relative % improvement (OH → BN)", fontsize=LABEL_FS - 1)
    cbar.ax.tick_params(labelsize=TICK_FS - 1)

    fig.suptitle(
        "Relative % improvement: OH (onehot) → BN (binary) encoding\n"
        f"task batch size={TASK_BATCH_SIZE},  points per batch={POINTS_PER_BATCH},  "
        f"iters={ITERS},  warmup iters={WARMUP_ITERS}",
        fontsize=13,
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Heatmaps saved to: {out_path}")
    # Also save a PDF alongside the raster image
    pdf_path = out_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Heatmaps saved to: {pdf_path}")


def main():
    # --- 1. Run sweep ---
    total = len(D_VALUES) * len(K_VALUES)
    print(f"Sweeping {total} (d,k) combinations …")
    for idx, (d, k) in enumerate(itertools.product(D_VALUES, K_VALUES), 1):
        print(f"[{idx:2d}/{total}]", end=" ")
        run_one(d, k)

    # --- 2. Load all results ---
    print("\nLoading results …")
    df = load_results()

    # --- 3. Plot ---
    print("Plotting heatmaps …")
    plot_heatmaps(df, out_path=f"{OUT_DIR}/heatmaps.png")  # also produces heatmaps.pdf


if __name__ == "__main__":
    main()
