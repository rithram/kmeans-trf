#!/usr/bin/env python

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from kmt import KModel
from losses import SoftKMObj, kmeans_obj_batched
from plot_eval_ood_gen import get_scratch

np.set_printoptions(precision=2)
pd.set_option("display.precision", 2)
print(f"Cuda available: {torch.cuda.is_available()}")
GPUE = torch.cuda.is_available()


def print_gpu_stats(state: str) -> None:
    if not torch.cuda.is_available():
        return
    # Source - https://stackoverflow.com/a
    # Posted by prosti, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-11-07, License - CC BY-SA 4.0

    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print(f"{state}: reserved {r * 100 // t}%, allocated {a * 100 // t}%, free {f * 100 // t}%")
    f, t = torch.cuda.mem_get_info()
    print(f"{state}: free {f * 100 // t}%")


class PerturbedWeights:
    def __init__(
        self,
        origin: list[torch.Tensor],
        xdir: list[torch.Tensor],
        ydir: list[torch.Tensor],
    ) -> None:
        self.origin = origin
        self.xdir = xdir
        self.ydir = ydir

    def set_weights(self, M: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.nn.Module:
        for m, o, dx, dy in zip(M.parameters(), self.origin, self.xdir, self.ydir, strict=True):
            m.data = o + (x * dx) + (y * dy)
        return M

    def reset_weights(self, M: torch.nn.Module) -> torch.nn.Module:
        for m, o in zip(M.parameters(), self.origin, strict=True):
            m.data = o
        return M


def write_to_file(lsurf: list[tuple], cols: list[str], ofile: str, b1: bool) -> None:
    if b1:
        pd.DataFrame(lsurf, columns=cols).to_csv(ofile, header=True, index=False)
    else:
        pd.DataFrame(lsurf, columns=cols).to_csv(ofile, mode="a", header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        help="Model checkpoint",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output",
        help="Directory for output",
        type=str,
        default="",
    )
    parser.add_argument("--aub", help="Max absolute range", type=float, default=1.0)
    parser.add_argument("--nsteps", help="Number of values per axis", type=int, default=10)
    parser.add_argument(
        "--pseed",
        help="Seed for picking projection directions",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--wbsz",
        help="Batch size for appending",
        type=int,
        default=100,
    )

    args = parser.parse_args()
    assert os.path.exists(args.checkpoint)
    assert args.output == "" or os.path.exists(args.output)
    save_output = args.output != ""
    assert args.aub > 0.0
    assert args.nsteps > 2
    checkpoint = torch.load(args.checkpoint)
    cli_args = checkpoint["cli_args"]
    NDIMS = cli_args["ndims"]
    K = cli_args["nclusters"]
    SEED = cli_args["seed"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print(f"Model trained for d={NDIMS}, k={K}, seed={SEED}")
    # load model
    ssize = {
        "onehot": K,
        "binary": np.ceil(np.log2(K)).astype(int),
        "none": 0,
    }
    demb = NDIMS + ssize[cli_args["scratch"]]
    model = KModel(
        demb,
        cli_args["dqkv"] * demb,
        inv_temp=cli_args["attn_itemp"],
        dropout_p=cli_args["dropout"],
        act=cli_args["attn_act"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if GPUE:
        model = model.to("cuda")
    print_gpu_stats("model loaded")

    # processing output file
    ldf_prev = None
    if save_output:
        outfile = (
            args.checkpoint.split("/")[-1].replace("_last.pt", "")
            + f"__aub:{args.aub}__nsteps:{args.nsteps}__pseed:{args.pseed}__"
            + "lsurf.csv"
        )
        outfile = os.path.join(args.output, outfile)
        print(f"Saving the loss surface in {outfile}")
        if os.path.exists(outfile):
            ldf_prev = pd.read_csv(outfile)
            print(f"Found existing loss surface:\n{outfile}")
            print(f"Surface size: {ldf_prev.shape}")
            if (args.nsteps * args.nsteps) == ldf_prev.shape[0]:
                print("Loss surface computation seems complete -- EXITING")
                sys.exit(0)

    # Set seeds for weight perturbation directions
    torch.manual_seed(args.pseed)
    np.random.seed(args.pseed)
    random.seed(args.pseed)

    # setting up the 2 random directions to perturb on
    orps = []
    dir1 = []
    dir2 = []
    mm = []
    for p in model.parameters():
        with torch.no_grad():
            num = torch.linalg.norm(p).item()
            rnd1 = torch.randn_like(p)
            den1 = torch.linalg.norm(rnd1).item()
            rnd1 *= num / den1
            rnd2 = torch.randn_like(p)
            den2 = torch.linalg.norm(rnd2).item()
            rnd2 *= num / den2
            orps += [p.clone().detach()]
            mm += [p.clone().detach()]
            dir1 += [rnd1]
            dir2 += [rnd2]
    assert len(orps) == len(dir1) == len(dir2)
    pw = PerturbedWeights(orps, dir1, dir2)
    print_gpu_stats("post-perturbed weights")
    xyvals = torch.linspace(-args.aub, args.aub, steps=args.nsteps)
    print(f"Number of values per axis: {xyvals.size(0)}")

    # validation tasks
    val_tasks = checkpoint["val_tasks"]
    vv = [(XX.clone().detach(), CC.clone().detach()) for (XX, CC) in val_tasks]
    scratched_val_tasks = [
        (
            BX,
            get_scratch(
                BX,
                BC,
                cli_args["scratch"],
                ssize[cli_args["scratch"]],
            ),
        )
        for BX, BC in val_tasks
    ]

    # train-loss
    criterion = SoftKMObj(
        gamma=cli_args["loss_itemp"], act=cli_args["loss_act"], logloss=cli_args["logloss"]
    )

    def eval_model(
        m: torch.nn.Module,
        task_batches: list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m.eval()
        kmobjs, losses = [], []
        for BX, (BXX, BCC) in task_batches:
            _, out = m(BXX, BCC)
            centers = out[:, :, :NDIMS]
            kmobjs += [kmeans_obj_batched(BX, centers).to("cpu")]
            losses += [criterion(BX, centers).to("cpu")]
        return torch.concat(kmobjs), torch.concat(losses)

    # compute initial metrics
    print_gpu_stats("pre-first eval")
    start_time = time.time()
    with torch.no_grad():
        kmeans_objs_1, train_losses_1 = eval_model(model, scratched_val_tasks)
    print(kmeans_objs_1.shape, train_losses_1.shape)
    teval = time.time() - start_time
    print(
        f"Original metrics -- train loss:{train_losses_1.mean():.4f}, "
        f"kmeans obj:{kmeans_objs_1.mean():.4f}, time:{teval:.4f}"
    )
    # start_time = time.time()
    # with torch.no_grad():
    #   kmeans_objs_1, train_losses_1 = eval_model(model, val_tasks)
    # print(kmeans_objs_1.shape, train_losses_1.shape)
    # teval = time.time() - start_time
    # print(
    #   f"Original metrics -- train loss:{train_losses_1.mean():.4f}, "
    #   f"kmeans obj:{kmeans_objs_1.mean():.4f}, time:{teval:.4f}"
    # )
    print_gpu_stats("post-first eval")

    loss_surface = []
    lcols = ["idx", "xval", "yval", "train-loss", "kmeans-obj", "time"]
    done_idx = 0 if ldf_prev is None else ldf_prev["idx"].values.astype(np.int32)[-1]
    print(f"skipping first {done_idx} grid points ...")
    first_batch = True if done_idx == 0 else False
    eidx = 0
    for dx in tqdm(xyvals):
        for dy in xyvals:
            eidx += 1
            if eidx <= done_idx:
                continue
            model = pw.set_weights(model, dx, dy)
            start_time = time.time()
            with torch.no_grad():
                kmeans_objs, train_losses = eval_model(model, scratched_val_tasks)
            teval = time.time() - start_time
            loss_surface += [
                (eidx, dx.item(), dy.item(), train_losses.mean(), kmeans_objs.mean(), teval)
            ]
            if save_output and len(loss_surface) == args.wbsz:
                # batched appending instead of repeated writing
                write_to_file(loss_surface, lcols, outfile, first_batch)
                first_batch = False
                loss_surface = []
            model = pw.reset_weights(model)
            # start_time = time.time()
            # with torch.no_grad():
            #   kmeans_objs, train_losses = eval_model(model, val_tasks)
            # teval = time.time() - start_time
            # print(
            #   f"Original metrics -- train loss:{train_losses.mean():.4f}, "
            #   f"kmeans obj:{kmeans_objs.mean():.4f}, time:{teval:.4f}"
            # )
    if save_output and len(loss_surface) > 0:
        # appending/writing the final batch
        write_to_file(loss_surface, lcols, outfile, first_batch)
        first_batch = False
    ldf = pd.read_csv(outfile) if save_output else pd.DataFrame(loss_surface, columns=lcols)
    assert ldf.shape[0] == (args.nsteps * args.nsteps)
    print(f"Loss surface size: {ldf.shape}")
    model = pw.reset_weights(model)
    # compute metrics a final time
    start_time = time.time()
    with torch.no_grad():
        kmeans_objs, train_losses = eval_model(model, scratched_val_tasks)
    print(kmeans_objs.shape, train_losses.shape)
    teval = time.time() - start_time
    print(
        f"final metrics -- train loss:{train_losses.mean():.4f}, "
        f"kmeans obj:{kmeans_objs.mean():.4f}, time:{teval:.4f}"
    )

    for task_idx, ((X, C), (XX, CC)) in enumerate(zip(val_tasks, vv, strict=True)):
        assert torch.all(X == XX), f"Task {task_idx}: X check"
        assert torch.all(C == CC), f"Task {task_idx}: C check"
    print("task data check ... OK")

    nparam_mats = len(mm)
    for midx, (p, pp) in enumerate(zip(model.parameters(), mm, strict=True)):
        assert torch.all(p == pp), f"Model param idx: {midx} / {nparam_mats}"
    print("model param check ... OK")

    kdiff = kmeans_objs - kmeans_objs_1
    print(f"kmeans-objs diff range: {kdiff.min():.4f}, {kdiff.max():.4f}")
    ldiff = train_losses - train_losses_1
    print(f"train loss diff range: {ldiff.min():.4f}, {ldiff.max():.4f}")
