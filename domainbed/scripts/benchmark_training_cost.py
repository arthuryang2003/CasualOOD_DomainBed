# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Lightweight efficiency benchmark: number of (trainable) parameters and training
cost measured as wall-clock time per training update (ms / step).

Example:
python -m domainbed.scripts.benchmark_training_cost \
  --data_dir ./domainbed/data \
  --dataset PACS \
  --test_envs 0 \
  --methods ERM LFME ASGDRO VITA \
  --batch_size 32 \
  --output_dir ./benchmark_output
"""

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from domainbed import algorithms
from domainbed import datasets
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader


def _maybe_load_hparams_override(hparams_arg: Optional[str]) -> Dict[str, Any]:
    if not hparams_arg:
        return {}
    if os.path.isfile(hparams_arg):
        with open(hparams_arg, "r") as f:
            return json.load(f)
    return json.loads(hparams_arg)


def _collect_optimizers(algorithm: torch.nn.Module) -> List[torch.optim.Optimizer]:
    opts: List[torch.optim.Optimizer] = []
    for v in algorithm.__dict__.values():
        if isinstance(v, torch.optim.Optimizer):
            opts.append(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, torch.optim.Optimizer):
                    opts.append(item)
        elif isinstance(v, dict):
            for item in v.values():
                if isinstance(item, torch.optim.Optimizer):
                    opts.append(item)
    # Fallback: common attribute name(s)
    if not opts:
        maybe = getattr(algorithm, "optimizer", None)
        if isinstance(maybe, torch.optim.Optimizer):
            opts.append(maybe)
    return opts


def count_trainable_params(algorithm: torch.nn.Module) -> int:
    """
    Count parameters that would be updated by the algorithm's optimizer(s).

    Note: some algorithms in this repo keep modules in Python lists (not
    registered submodules). Optimizers still hold references to those
    parameters, so counting via optimizers is robust.
    """
    opts = _collect_optimizers(algorithm)
    if opts:
        seen: set[int] = set()
        params: List[torch.nn.Parameter] = []
        for opt in opts:
            for group in opt.param_groups:
                for p in group.get("params", []):
                    if p is None:
                        continue
                    pid = id(p)
                    if pid in seen:
                        continue
                    seen.add(pid)
                    if isinstance(p, torch.nn.Parameter) and p.requires_grad:
                        params.append(p)
        return int(sum(p.numel() for p in params))

    # Fallback for algorithms without discoverable optimizers
    return int(sum(p.numel() for p in algorithm.parameters() if p.requires_grad))


def _to_device_minibatches(
    batch: Sequence[Sequence[torch.Tensor]], device: torch.device
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    minibatches_device: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for mb in batch:
        if len(mb) < 2:
            raise ValueError("Expected minibatch to have at least (x, y).")
        x, y = mb[0], mb[1]
        minibatches_device.append((x.to(device, non_blocking=True), y.to(device, non_blocking=True)))
    return minibatches_device


def _make_train_loaders(
    dataset: datasets.MultipleDomainDataset,
    test_envs: Sequence[int],
    hparams: Dict[str, Any],
    *,
    holdout_fraction: float,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> List[InfiniteDataLoader]:
    train_env_indices = [i for i in range(len(dataset)) if i not in set(test_envs)]

    env_datasets = []
    for env_i in train_env_indices:
        env = dataset[env_i]
        if holdout_fraction > 0:
            _, in_ = misc.split_dataset(
                env,
                int(len(env) * holdout_fraction),
                misc.seed_hash(seed, env_i),
            )
            env_datasets.append(in_)
        else:
            env_datasets.append(env)

    train_loaders: List[InfiniteDataLoader] = []
    for env in env_datasets:
        weights = misc.make_weights_for_balanced_classes(env) if hparams.get("class_balanced", False) else None
        train_loaders.append(
            InfiniteDataLoader(
                dataset=env,
                weights=weights,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        )
    return train_loaders


@torch.no_grad()
def _device_sanity_print(device: torch.device) -> None:
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"Using CUDA device {idx}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("Using CPU")


def _benchmark_method(
    algorithm_name: str,
    dataset: datasets.MultipleDomainDataset,
    base_hparams: Dict[str, Any],
    *,
    train_loaders: Sequence[InfiniteDataLoader],
    test_envs: Sequence[int],
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> Dict[str, Any]:
    hparams = hparams_registry.default_hparams(algorithm_name, dataset.__class__.__name__)

    # Keep protocol-identical settings across methods.
    for k in ["data_augmentation", "resnet18", "vit", "dinov2"]:
        if k in base_hparams:
            hparams[k] = base_hparams[k]
    for k in ["lr", "weight_decay", "batch_size"]:
        if k in base_hparams:
            hparams[k] = base_hparams[k]
    if "no_pretrain" in base_hparams:
        hparams["no_pretrain"] = base_hparams["no_pretrain"]

    # Make VITA stay in the labeled training phase for the whole benchmark.
    if algorithm_name == "VITA":
        hparams["phase1_steps"] = int(warmup_steps + measure_steps + 10)
        hparams["phase2_steps"] = 0
        hparams["finetune_steps"] = 0

    algorithm_class = algorithms.get_algorithm_class(algorithm_name)
    num_train_domains = len(dataset) - len(set(test_envs))
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, num_train_domains, hparams)
    algorithm.to(device)
    algorithm.train()

    params = count_trainable_params(algorithm)

    train_minibatches_iterator = zip(*train_loaders)

    def next_minibatches() -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batch = next(train_minibatches_iterator)
        return _to_device_minibatches(batch, device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup_steps):
        minibatches_device = next_minibatches()
        algorithm.update(minibatches_device, None)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Measure
    times_ms: List[float] = []
    for _ in range(measure_steps):
        minibatches_device = next_minibatches()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        algorithm.update(minibatches_device, None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times = np.asarray(times_ms, dtype=np.float64)
    return {
        "method": algorithm_name,
        "params": int(params),
        "train_step_time_mean_ms": float(times.mean()),
        "train_step_time_std_ms": float(times.std(ddof=1) if len(times) > 1 else 0.0),
    }


def _write_csv(rows: Sequence[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "params", "train_step_time_mean_ms", "train_step_time_std_ms"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_params_m(params: int) -> str:
    return f"{params / 1e6:.2f}M"


def _write_latex_table(rows: Sequence[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def method_label(m: str) -> str:
        if m == "VITA":
            return "VITA (ours)"
        return m

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & \#Params & Train time (ms/step) \\")
    lines.append(r"\midrule")
    for row in rows:
        m = method_label(str(row["method"]))
        params = _fmt_params_m(int(row["params"]))
        mean_ms = float(row["train_step_time_mean_ms"])
        std_ms = float(row["train_step_time_std_ms"])
        lines.append(rf"{m} & {params} & {mean_ms:.1f} $\pm$ {std_ms:.1f} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{-1mm}")
    lines.append(r"\caption{Efficiency comparison (same dataset/backbone/batch size/hardware).}")
    lines.append(r"\end{table}")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark #params and ms/step for DG algorithms")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--test_envs", type=int, nargs="+", default=[0])
    parser.add_argument("--methods", type=str, nargs="+", default=["ERM", "LFME", "ASGDRO", "VITA"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--backbone", type=str, choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--data_augmentation", action="store_true", default=False)
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--measure_steps", type=int, default=100)
    parser.add_argument("--holdout_fraction", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hparams",
        type=str,
        default=None,
        help="JSON string or path to JSON file; applied to all methods.",
    )
    parser.add_argument("--output_dir", type=str, default="benchmark_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _device_sanity_print(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Shared hparams used to (1) build the dataset and (2) enforce consistency across methods.
    base_hparams = hparams_registry.default_hparams("ERM", args.dataset)
    base_hparams.update(_maybe_load_hparams_override(args.hparams))
    base_hparams["data_augmentation"] = bool(args.data_augmentation)
    base_hparams["resnet18"] = (args.backbone == "resnet18")
    base_hparams["no_pretrain"] = (not bool(args.pretrained))
    base_hparams["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        base_hparams["lr"] = float(args.lr)
    if args.weight_decay is not None:
        base_hparams["weight_decay"] = float(args.weight_decay)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, base_hparams)
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    if device.type != "cuda" and any(m == "LFME" for m in args.methods):
        raise RuntimeError("LFME in this repo assumes CUDA; run with a GPU or remove LFME from --methods.")

    train_loaders = _make_train_loaders(
        dataset,
        args.test_envs,
        base_hparams,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=(args.num_workers if args.num_workers is not None else dataset.N_WORKERS),
    )

    results: List[Dict[str, Any]] = []
    for method in args.methods:
        print(f"\nBenchmarking {method} ...")
        row = _benchmark_method(
            method,
            dataset,
            base_hparams,
            train_loaders=train_loaders,
            test_envs=args.test_envs,
            device=device,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )
        print(
            f"  params={row['params']:,} | "
            f"train_step_time={row['train_step_time_mean_ms']:.2f}±{row['train_step_time_std_ms']:.2f} ms"
        )
        results.append(row)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = os.path.join(args.output_dir, "training_cost.csv")
    tex_path = os.path.join(args.output_dir, "training_cost_table.tex")
    _write_csv(results, csv_path)
    _write_latex_table(results, tex_path)

    print(f"\nWrote CSV: {csv_path}")
    print(f"Wrote LaTeX: {tex_path} (requires \\usepackage{{booktabs}})")


if __name__ == "__main__":
    main()
