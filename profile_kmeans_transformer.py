#!/usr/bin/env python

import argparse
import random
import time

import numpy as np
import pandas as pd
import torch

from ctasks import ClusteringTasks
from kmt import KModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile forward/backward runtime and memory for the k-means transformer "
            "with onehot vs binary scratch encodings."
        )
    )
    parser.add_argument("--d", type=int, required=True, help="Data dimensionality.")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters.")
    parser.add_argument(
        "--task-batch-size", type=int, required=True, help="Number of clustering tasks per batch."
    )
    parser.add_argument(
        "--points-per-batch", type=int, required=True, help="Number of points per task."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to profile on. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--iters", type=int, default=20, help="Number of timed profiling iterations."
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=5, help="Number of warmup iterations before timing."
    )
    parser.add_argument("--dqkv", type=int, default=1, help="Projection dimensionality multiplier.")
    parser.add_argument(
        "--attn-itemp", type=float, default=1.0, help="Inverse temperature for attention."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability. Use 0.0 for stable profiling.",
    )
    parser.add_argument(
        "--attn-act",
        choices=["softmax", "sparsemax"],
        default="softmax",
        help="Attention activation.",
    )
    parser.add_argument("--dist", type=str, default="normal", help="Task distribution name.")
    parser.add_argument("--scale", type=float, default=0.1, help="Distribution scale.")
    parser.add_argument("--seed", type=int, default=548977, help="Random seed.")
    parser.add_argument(
        "--csv", type=str, default="", help="Optional path to save profiling results as CSV."
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.05,
        help="Quantile level q for reporting q and (1-q) quantiles.",
    )
    parser.add_argument(
        "--random-encoding",
        action="store_true",
        help="Fill cluster index encoding with random values instead of structured patterns.",
    )
    parser.add_argument(
        "--random-data",
        action="store_true",
        help=(
            "Skip structured data generation entirely. Instead, use raw random "
            "(task_batch_size x points_per_batch x demb) tensors for both xx and cc, "
            "where demb = d + scratch_size."
        ),
    )
    args = parser.parse_args()

    assert args.d >= 2
    assert args.k >= 2
    assert args.task_batch_size >= 1
    assert args.points_per_batch >= args.k
    assert args.iters >= 1
    assert args.warmup_iters >= 0
    assert args.dqkv >= 1
    assert args.attn_itemp > 0.0
    assert args.scale > 0.0
    assert 0.0 < args.quantile < 0.5
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scratch_size(scratch: str, k: int) -> int:
    if scratch == "onehot":
        return k
    if scratch == "binary":
        return int(np.ceil(np.log2(k)))
    raise ValueError(f"Unsupported scratch encoding: {scratch}")


def get_cluster_id_enc(
    scratch: str, k: int, device: torch.device, random_encoding: bool = False
) -> torch.Tensor:
    ssize = scratch_size(scratch, k)

    if random_encoding:
        # Fill with random values instead of structured patterns
        return torch.randn(k, ssize, device=device, dtype=torch.float32)

    if scratch == "onehot":
        return torch.eye(ssize, device=device)
    assert scratch == "binary"
    cols = []
    ids = torch.arange(k, device=device)
    for _ in range(ssize):
        cols.append(ids % 2)
        ids = ids // 2
    return torch.stack(cols, dim=1).to(torch.float32)


def build_model(args: argparse.Namespace, scratch: str) -> tuple[KModel, int, int]:
    ssize = scratch_size(scratch, args.k)
    demb = args.d + ssize
    model = KModel(
        demb,
        args.dqkv * demb,
        inv_temp=args.attn_itemp,
        dropout_p=args.dropout,
        act=args.attn_act,
    )
    return model, ssize, demb


def prepare_inputs(
    args: argparse.Namespace,
    scratch: str,
    device: torch.device,
    random_encoding: bool = False,
    random_data: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    ssize = scratch_size(scratch, args.k)
    demb = args.d + ssize

    if random_data:
        xx = torch.randn(
            args.task_batch_size, args.points_per_batch, demb, device=device, dtype=torch.float32
        )
        cc = torch.randn(args.task_batch_size, args.k, demb, device=device, dtype=torch.float32)
        return xx, cc

    task_gen = ClusteringTasks(
        args.k,
        args.k,
        args.d,
        args.points_per_batch,
        args.points_per_batch,
        [args.dist],
        scale=args.scale,
        equal_mix=True,
        equal_scales=True,
    )
    samples, centers = task_gen.sample_batch(args.task_batch_size, same_dist_batch=True)
    samples = samples.to(device=device, dtype=torch.float32)
    centers = centers.to(device=device, dtype=torch.float32)

    zeros = torch.zeros(
        args.task_batch_size, args.points_per_batch, ssize, device=device, dtype=samples.dtype
    )
    enc = (
        get_cluster_id_enc(scratch, args.k, device, random_encoding)
        .unsqueeze(0)
        .repeat(args.task_batch_size, 1, 1)
    )
    xx = torch.cat((samples.detach(), zeros), dim=2)
    cc = torch.cat((centers.detach(), enc), dim=2)
    return xx, cc


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# Memory is only tracked on CUDA; on CPU the memory metrics are reported as NaN.
def current_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024**2)
    return float("nan")


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    return float("nan")


def summarize_metric(df: pd.DataFrame, column: str, quantile: float) -> dict[str, float]:
    values = df[column].to_numpy(dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    q_low = float(np.quantile(values, quantile))
    q_high = float(np.quantile(values, 1.0 - quantile))
    return {
        f"{column}_mean": mean,
        f"{column}_std": std,
        f"{column}_q_low": q_low,
        f"{column}_q_high": q_high,
    }


def profile_encoding(
    args: argparse.Namespace, scratch: str, device: torch.device, train_mode: bool = True
) -> dict:
    model, ssize, demb = build_model(args, scratch)
    model = model.to(device)
    if train_mode:
        model.train()
    else:
        model.eval()

    xx_base, cc_base = prepare_inputs(args, scratch, device, args.random_encoding, args.random_data)

    # KModel contains SAX (self-attention over XX: shape tbs x npb x npb) and SAC
    # (self-attention over CC: shape tbs x k x k).  Both the cdist distance matrix
    # and the softmax output for SAX scale as tbs x npb x npb, independent of the
    # encoding (demb does not appear in those shapes).  They form a fixed floor in
    # the peak measurement that is identical for onehot and binary and obscures the
    # encoding-driven difference.  We subtract them analytically so that
    # enc_delta_mb isolates only the encoding-sensitive memory cost.
    # cdist output + softmax output for SAX: 2 x (tbs x npb x npb) float32
    sax_floor_mb = (
        2 * args.task_batch_size * args.points_per_batch * args.points_per_batch * 4 / (1024**2)
    )

    def run_step() -> dict:
        # Free any lingering gradient buffers from the previous step before
        # taking the baseline snapshot, so start_mem is a clean pre-step floor.
        model.zero_grad(set_to_none=True)

        sync_if_needed(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start_mem = current_memory_mb(device)

        # Clones are allocated after the baseline so they are included in the
        # peak window and in the incremental cost.
        xx = xx_base.detach().clone().requires_grad_(train_mode)
        cc = cc_base.detach().clone().requires_grad_(train_mode)

        start = time.perf_counter()

        if train_mode:
            xxx, ccc = model(xx, cc)
        else:
            with torch.no_grad():
                xxx, ccc = model(xx, cc)

        sync_if_needed(device)
        forward_end = time.perf_counter()

        result: dict = {"forward_ms": (forward_end - start) * 1000.0}

        if train_mode:
            loss = xxx[:, :, : args.d].square().mean() + ccc[:, :, : args.d].square().mean()
            loss.backward()

            sync_if_needed(device)
            end = time.perf_counter()
            result["backward_ms"] = (end - forward_end) * 1000.0
            result["total_ms"] = (end - start) * 1000.0
            result["loss"] = loss.item()

        peak_mem = peak_memory_mb(device)
        peak_delta = peak_mem - start_mem
        result["peak_delta_mb"] = peak_delta
        result["enc_delta_mb"] = peak_delta - sax_floor_mb

        return result

    for _ in range(args.warmup_iters):
        run_step()

    rows = [run_step() for _ in range(args.iters)]
    df = pd.DataFrame(rows)

    summary = {
        "encoding": scratch,
        "mode": "train" if train_mode else "eval",
        "device": device.type,
        "model": type(model).__name__,
        "d": args.d,
        "k": args.k,
        "task_batch_size": args.task_batch_size,
        "points_per_batch": args.points_per_batch,
        "scratch_size": ssize,
        "embedding_dim": demb,
        "iters": args.iters,
        "quantile": args.quantile,
    }
    train_metrics = ["backward_ms", "total_ms"] if train_mode else []
    for metric in ["forward_ms"] + train_metrics + ["peak_delta_mb", "enc_delta_mb"]:
        summary.update(summarize_metric(df, metric, args.quantile))
    summary["peak_delta_mb_max"] = float(df["peak_delta_mb"].max())
    summary["enc_delta_mb_max"] = float(df["enc_delta_mb"].max())
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    results = [
        profile_encoding(args, "onehot", device, train_mode=True),
        profile_encoding(args, "binary", device, train_mode=True),
        profile_encoding(args, "onehot", device, train_mode=False),
        profile_encoding(args, "binary", device, train_mode=False),
    ]
    df = pd.DataFrame(results)

    col_map = {
        "encoding": "enc",
        "mode": "mode",
        "device": "dev",
        "model": "mdl",
        "task_batch_size": "tbs",
        "points_per_batch": "npb",
        "scratch_size": "ssz",
        "embedding_dim": "emb",
        "iters": "n",
        "quantile": "q",
        "forward_ms_mean": "f_ms",
        "forward_ms_std": "f_sd",
        "forward_ms_q_low": "f_lo",
        "forward_ms_q_high": "f_hi",
        "backward_ms_mean": "b_ms",
        "backward_ms_std": "b_sd",
        "backward_ms_q_low": "b_lo",
        "backward_ms_q_high": "b_hi",
        "total_ms_mean": "t_ms",
        "total_ms_std": "t_sd",
        "total_ms_q_low": "t_lo",
        "total_ms_q_high": "t_hi",
        "peak_delta_mb_mean": "pdmem_ms",
        "peak_delta_mb_std": "pdmem_sd",
        "peak_delta_mb_q_low": "pdmem_lo",
        "peak_delta_mb_q_high": "pdmem_hi",
        "peak_delta_mb_max": "pdmem_max",
        "enc_delta_mb_mean": "edmem_ms",
        "enc_delta_mb_std": "edmem_sd",
        "enc_delta_mb_q_low": "edmem_lo",
        "enc_delta_mb_q_high": "edmem_hi",
        "enc_delta_mb_max": "edmem_max",
    }
    out_df = df.rename(columns=col_map)

    legend = {
        "enc": "encoding",
        "mode": "train/eval mode",
        "dev": "device",
        "mdl": "model",
        "tbs": "task_batch_size",
        "npb": "points_per_batch",
        "ssz": "scratch_size",
        "emb": "embedding_dim",
        "n": "iters",
        "q": "quantile",
        "f_ms": "forward_ms_mean",
        "f_sd": "forward_ms_std",
        "f_lo": "forward_ms_q_low",
        "f_hi": "forward_ms_q_high",
        "b_ms": "backward_ms_mean  (train only)",
        "b_sd": "backward_ms_std   (train only)",
        "b_lo": "backward_ms_q_low (train only)",
        "b_hi": "backward_ms_q_high(train only)",
        "t_ms": "total_ms_mean     (train only)",
        "t_sd": "total_ms_std      (train only)",
        "t_lo": "total_ms_q_low    (train only)",
        "t_hi": "total_ms_q_high   (train only)",
        "pdmem_ms": "peak_delta_mb_mean",
        "pdmem_sd": "peak_delta_mb_std",
        "pdmem_lo": "peak_delta_mb_q_low",
        "pdmem_hi": "peak_delta_mb_q_high",
        "pdmem_max": "peak_delta_mb_max",
        "edmem_ms": "enc_delta_mb_mean",
        "edmem_sd": "enc_delta_mb_std",
        "edmem_lo": "enc_delta_mb_q_low",
        "edmem_hi": "enc_delta_mb_q_high",
        "edmem_max": "enc_delta_mb_max",
    }

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("Column legend:")
    for short_name, long_name in legend.items():
        print(f"  {short_name:>8} = {long_name}")
    print()
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if args.csv:
        out_df.to_csv(args.csv, index=False)
        print(f"\nSaved results to {args.csv}")


if __name__ == "__main__":
    main()
