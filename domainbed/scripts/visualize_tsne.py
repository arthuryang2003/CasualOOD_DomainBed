"""
Example:
    python -m domainbed.scripts.visualize_tsne \
      --data_dir=./domainbed/data \
      --input_dir=./sweep/CelebA_Blond/VITA_v2_test2 \
      --dataset=CelebA_Blond \
      --algorithm=VITA \
      --test_env=2
"""

import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from domainbed import algorithms, datasets, hparams_registry, model_selection
from domainbed.lib import misc, reporting


BEST_CHECKPOINT_NAMES = (
    "model_best.pkl",
    "best.pt",
    "best.pth",
    "checkpoint_best.pth",
    "best_model.pth",
)

LAST_CHECKPOINT_NAMES = (
    "model.pkl",
    "last.pt",
    "last.pth",
    "checkpoint_last.pth",
    "model_last.pth",
)

RESULT_FILE_NAMES = (
    "results.jsonl",
    "results.json",
    "results.txt",
    "metrics.json",
    "summary.csv",
    "done",
)

PREFERRED_VAL_KEYS = (
    "val_acc",
    "validation_acc",
    "best_val_acc",
    "env{test_env}_out_acc",
    "env{test_env}_val_acc",
    "test_out_acc",
)

FALLBACK_TEST_KEYS = (
    "test_acc",
    "best_test_acc",
    "env{test_env}_in_acc",
    "env{test_env}_test_acc",
    "test_in_acc",
)


@dataclass
class CheckpointSelection:
    checkpoint_path: str
    checkpoint_name: str
    run_dir: str
    selection_reason: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    metric_source: Optional[str] = None


def print_candidates(label: str, paths: Sequence[str]) -> None:
    return


def candidate_checkpoint_paths(root_dir: str, names: Sequence[str]) -> List[str]:
    candidates = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in names:
            if name in filenames:
                candidates.append(os.path.join(dirpath, name))
    return sorted(set(candidates))


def candidate_metric_paths(root_dir: str) -> List[str]:
    candidates = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in RESULT_FILE_NAMES:
            if name in filenames:
                candidates.append(os.path.join(dirpath, name))
    return sorted(set(candidates))


def parse_float(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def flatten_numeric_entries(data, prefix: str = "") -> Dict[str, float]:
    values: Dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            values.update(flatten_numeric_entries(value, next_prefix))
    elif isinstance(data, list):
        if data and all(isinstance(item, dict) for item in data):
            return values
        for idx, value in enumerate(data):
            next_prefix = f"{prefix}[{idx}]"
            values.update(flatten_numeric_entries(value, next_prefix))
    else:
        numeric = parse_float(data)
        if numeric is not None and prefix:
            values[prefix] = numeric
    return values


def resolve_metric_candidates(test_env: int) -> Tuple[List[str], List[str]]:
    preferred = [key.format(test_env=test_env) for key in PREFERRED_VAL_KEYS]
    fallback = [key.format(test_env=test_env) for key in FALLBACK_TEST_KEYS]
    return preferred, fallback


def choose_metric_from_rows(rows: Sequence[Dict[str, object]], test_env: int) -> Optional[Tuple[str, float]]:
    if not rows:
        return None

    preferred, fallback = resolve_metric_candidates(test_env)
    available_keys = set()
    for row in rows:
        available_keys.update(row.keys())

    for key in preferred:
        if key in available_keys:
            best = [parse_float(row.get(key)) for row in rows]
            best = [value for value in best if value is not None]
            if best:
                return key, max(best)

    for key in fallback:
        if key in available_keys:
            best = [parse_float(row.get(key)) for row in rows]
            best = [value for value in best if value is not None]
            if best:
                return key, max(best)

    for row in reversed(rows):
        numeric_entries = {key: parse_float(value) for key, value in row.items()}
        numeric_entries = {key: value for key, value in numeric_entries.items() if value is not None}
        for key, value in numeric_entries.items():
            lowered = key.lower()
            if "val" in lowered and "acc" in lowered:
                return key, value
        for key, value in numeric_entries.items():
            lowered = key.lower()
            if "test" in lowered and "acc" in lowered:
                return key, value

    return None


def parse_jsonl_metrics(path: str, test_env: int) -> Optional[Tuple[str, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return choose_metric_from_rows(rows, test_env)


def parse_json_metrics(path: str, test_env: int) -> Optional[Tuple[str, float]]:
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            return None
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return choose_metric_from_rows(data, test_env)
    if isinstance(data, dict):
        numeric_entries = flatten_numeric_entries(data)
        rows = [numeric_entries] if numeric_entries else []
        return choose_metric_from_rows(rows, test_env)
    return None


def parse_csv_metrics(path: str, test_env: int) -> Optional[Tuple[str, float]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    return choose_metric_from_rows(rows, test_env)


def parse_text_metrics(path: str, test_env: int) -> Optional[Tuple[str, float]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        content = handle.read()
    pairs = re.findall(r"([A-Za-z0-9_./-]+)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", content)
    if not pairs:
        return None
    row = {key: float(value) for key, value in pairs}
    return choose_metric_from_rows([row], test_env)


def parse_metric_file(path: str, test_env: int) -> Optional[Tuple[str, float]]:
    name = os.path.basename(path)
    try:
        if name == "results.jsonl":
            return parse_jsonl_metrics(path, test_env)
        if name.endswith(".json"):
            return parse_json_metrics(path, test_env)
        if name.endswith(".csv"):
            return parse_csv_metrics(path, test_env)
        return parse_text_metrics(path, test_env)
    except OSError:
        return None


def score_run_dir(run_dir: str, test_env: int) -> Optional[Tuple[str, float, str]]:
    best_metric = None
    best_priority = -1
    preferred, fallback = resolve_metric_candidates(test_env)
    priority_map = {key: 3 for key in preferred}
    priority_map.update({key: 2 for key in fallback})

    for name in RESULT_FILE_NAMES:
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            continue
        parsed = parse_metric_file(path, test_env)
        if parsed is None:
            continue
        metric_name, metric_value = parsed
        priority = priority_map.get(metric_name, 1)
        if best_metric is None or priority > best_priority or (
            priority == best_priority and metric_value > best_metric[1]
        ):
            best_metric = (metric_name, metric_value, path)
            best_priority = priority

    return best_metric


def infer_run_dirs(input_dir: str) -> List[str]:
    run_dirs = []
    if any(os.path.isfile(os.path.join(input_dir, name)) for name in BEST_CHECKPOINT_NAMES + LAST_CHECKPOINT_NAMES):
        run_dirs.append(input_dir)
    immediate_subdirs = [
        os.path.join(input_dir, entry)
        for entry in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, entry))
    ]
    for subdir in immediate_subdirs:
        if any(os.path.isfile(os.path.join(subdir, name)) for name in BEST_CHECKPOINT_NAMES + LAST_CHECKPOINT_NAMES):
            run_dirs.append(subdir)
            continue
        if any(os.path.isfile(os.path.join(subdir, name)) for name in RESULT_FILE_NAMES):
            run_dirs.append(subdir)
    if not run_dirs:
        run_dirs.append(input_dir)
    return sorted(set(run_dirs))


def select_checkpoint_in_run(run_dir: str) -> Optional[str]:
    for name in BEST_CHECKPOINT_NAMES:
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    for name in LAST_CHECKPOINT_NAMES:
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    return None


def select_checkpoint_from_input_dir(
    input_dir: str, dataset: str, algorithm: str, test_env: int
) -> CheckpointSelection:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    best_ckpt_candidates = candidate_checkpoint_paths(input_dir, BEST_CHECKPOINT_NAMES)
    last_ckpt_candidates = candidate_checkpoint_paths(input_dir, LAST_CHECKPOINT_NAMES)
    metric_candidates = candidate_metric_paths(input_dir)
    print_candidates("Scanned best checkpoint candidates", best_ckpt_candidates)
    print_candidates("Scanned last checkpoint candidates", last_ckpt_candidates)
    print_candidates("Scanned metric file candidates", metric_candidates)

    direct_best = [path for path in best_ckpt_candidates if os.path.dirname(path) == input_dir]
    if direct_best:
        chosen = direct_best[0]
        return CheckpointSelection(
            checkpoint_path=chosen,
            checkpoint_name=os.path.basename(chosen),
            run_dir=os.path.dirname(chosen),
            selection_reason="direct best checkpoint file found in input_dir",
        )

    sweep_has_records = False
    for entry in os.listdir(input_dir):
        subdir = os.path.join(input_dir, entry)
        if not os.path.isdir(subdir):
            continue
        if os.path.isfile(os.path.join(subdir, "done")) and os.path.isfile(os.path.join(subdir, "results.jsonl")):
            sweep_has_records = True
            break

    if sweep_has_records:
        try:
            records = reporting.load_records(input_dir)
            grouped = reporting.get_grouped_records(records).filter(
                lambda r: r["dataset"] == dataset
                and r["algorithm"] == algorithm
                and r["test_env"] == test_env
            )
        except OSError:
            grouped = []
        if len(grouped):
            best_dir = None
            best_val = -float("inf")
            best_key = None
            for group in grouped:
                hparams_accs = model_selection.OracleSelectionMethod.hparams_accs(group["records"])
                if not hparams_accs:
                    continue
                run_acc, run_records = hparams_accs[0]
                if run_acc["val_acc"] > best_val:
                    recorded_dir = run_records[0]["args"]["output_dir"]
                    resolved_dir = recorded_dir
                    if not os.path.isabs(resolved_dir):
                        resolved_dir = os.path.normpath(os.path.join(input_dir, resolved_dir))
                    if not os.path.isdir(resolved_dir) and os.path.isdir(recorded_dir):
                        resolved_dir = recorded_dir
                    best_val = run_acc["val_acc"]
                    best_key = "val_acc"
                    best_dir = resolved_dir
            if best_dir:
                checkpoint_path = select_checkpoint_in_run(best_dir)
                if checkpoint_path:
                    return CheckpointSelection(
                        checkpoint_path=checkpoint_path,
                        checkpoint_name=os.path.basename(checkpoint_path),
                        run_dir=best_dir,
                        selection_reason="best run selected from results.jsonl sweep records",
                        metric_name=best_key,
                        metric_value=best_val,
                        metric_source=os.path.join(best_dir, "results.jsonl"),
                    )

    run_dirs = infer_run_dirs(input_dir)
    scored_runs = []
    fallback_runs = []
    for run_dir in run_dirs:
        checkpoint_path = select_checkpoint_in_run(run_dir)
        if checkpoint_path is None:
            continue
        metric = score_run_dir(run_dir, test_env)
        if metric is None:
            fallback_runs.append((run_dir, checkpoint_path))
            continue
        metric_name, metric_value, metric_source = metric
        scored_runs.append((metric_value, metric_name, metric_source, run_dir, checkpoint_path))

    if scored_runs:
        scored_runs.sort(key=lambda item: item[0], reverse=True)
        metric_value, metric_name, metric_source, run_dir, checkpoint_path = scored_runs[0]
        checkpoint_is_best = os.path.basename(checkpoint_path) in BEST_CHECKPOINT_NAMES
        selection_reason = "best checkpoint file found in selected run"
        if not checkpoint_is_best:
            selection_reason = "metric-selected run; falling back to last checkpoint in that run"
        return CheckpointSelection(
            checkpoint_path=checkpoint_path,
            checkpoint_name=os.path.basename(checkpoint_path),
            run_dir=run_dir,
            selection_reason=selection_reason,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_source=metric_source,
        )

    if fallback_runs:
        run_dir, checkpoint_path = fallback_runs[0]
        print("Warning: no usable metric file found; falling back to an available checkpoint.")
        reason = "fallback to available checkpoint because no best metric file was found"
        if os.path.basename(checkpoint_path) in BEST_CHECKPOINT_NAMES:
            reason = "best checkpoint file found but no metric file could be parsed"
        return CheckpointSelection(
            checkpoint_path=checkpoint_path,
            checkpoint_name=os.path.basename(checkpoint_path),
            run_dir=run_dir,
            selection_reason=reason,
        )

    raise RuntimeError(f"Could not find any usable checkpoint under {input_dir}")


def extract_state_dict(raw_obj) -> Dict[str, torch.Tensor]:
    if isinstance(raw_obj, dict):
        for key in ("model_dict", "state_dict", "model", "network"):
            value = raw_obj.get(key)
            if isinstance(value, dict):
                return value
        if raw_obj and all(torch.is_tensor(value) for value in raw_obj.values()):
            return raw_obj
    raise RuntimeError("Checkpoint does not contain a recognizable state dict.")


def load_checkpoint_bundle(checkpoint_path: str) -> Dict[str, object]:
    bundle = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(bundle, dict) and {"args", "model_input_shape", "model_num_classes", "model_num_domains", "model_hparams", "model_dict"} <= bundle.keys():
        return bundle

    sidecar = os.path.join(os.path.dirname(checkpoint_path), "model.pkl")
    if os.path.abspath(sidecar) == os.path.abspath(checkpoint_path):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    if not os.path.isfile(sidecar):
        raise RuntimeError(
            f"{checkpoint_path} does not include DomainBed metadata and no sidecar model.pkl was found."
        )
    metadata = torch.load(sidecar, map_location="cpu")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Sidecar metadata is invalid: {sidecar}")
    metadata = dict(metadata)
    metadata["model_dict"] = extract_state_dict(bundle)
    return metadata


def load_model_from_bundle(bundle: Dict[str, object]) -> algorithms.Algorithm:
    alg_name = bundle["args"]["algorithm"]
    alg_class = algorithms.get_algorithm_class(alg_name)
    model = alg_class(
        bundle["model_input_shape"],
        bundle["model_num_classes"],
        bundle["model_num_domains"],
        bundle["model_hparams"],
    )
    model.load_state_dict(bundle["model_dict"])
    return model


def unwrap_batch(batch, fallback_env: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (list, tuple)):
        raise ValueError("Expected dataloader batch to be a tuple/list.")
    if len(batch) < 2:
        raise ValueError("Expected batch to include at least (x, y).")

    x = batch[0]
    y = batch[1]
    if len(batch) >= 3:
        env = batch[2]
        if not torch.is_tensor(env):
            env = torch.as_tensor(env)
    else:
        env = torch.full((x.size(0),), fallback_env, dtype=torch.long)

    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    y = y.long()
    env = env.long()
    return x, y, env


def infer_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconstruct_target_split(
    dataset_obj,
    run_args: Dict[str, object],
    algorithm_name: str,
    test_env: int,
    split: str,
):
    env_dataset = dataset_obj[test_env]
    holdout_fraction = float(run_args.get("holdout_fraction", 0.2))
    trial_seed = int(run_args.get("trial_seed", 0))
    uda_holdout_fraction = float(run_args.get("uda_holdout_fraction", 0.0))
    task = run_args.get("task", "domain_generalization")

    if algorithm_name == "VITA" and uda_holdout_fraction <= 0:
        uda_holdout_fraction = 0.2

    out_split, in_split = misc.split_dataset(
        env_dataset,
        int(len(env_dataset) * holdout_fraction),
        misc.seed_hash(trial_seed, test_env),
    )

    uda_split = []
    if task == "domain_adaptation" or algorithm_name == "VITA" or uda_holdout_fraction > 0:
        uda_count = int(len(in_split) * uda_holdout_fraction)
        if uda_count > 0:
            uda_split, in_split = misc.split_dataset(
                in_split,
                uda_count,
                misc.seed_hash(trial_seed, test_env),
            )

    if split == "test":
        return in_split
    if split == "val":
        return out_split
    if split == "train":
        if len(uda_split):
            return uda_split
        print("Warning: requested --split=train but no target-domain train/uda split exists; using test split instead.")
        return in_split
    raise ValueError(f"Unsupported split: {split}")


def extract_zu_zs(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    if hasattr(model, "encode"):
        encoded = model.encode(x)
        if isinstance(encoded, dict) and "zu" in encoded and "zs" in encoded:
            return {"zu": encoded["zu"], "zs": encoded["zs"]}
        if isinstance(encoded, (tuple, list)) and len(encoded) >= 2:
            return {"zu": encoded[0], "zs": encoded[1]}
    raise RuntimeError(
        "Could not extract zu/zs from model.encode(x). "
        "This algorithm does not expose encode() features in a compatible way, "
        "so visualization is unavailable without modifying the algorithm."
    )


@torch.no_grad()
def collect_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    fallback_env: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_zu, all_zs, all_y, all_env = [], [], [], []
    model.eval()
    for batch in loader:
        x, y, env = unwrap_batch(batch, fallback_env=fallback_env)
        x = x.to(device)
        feats = extract_zu_zs(model, x)
        zu = feats["zu"].detach().cpu().numpy()
        zs = feats["zs"].detach().cpu().numpy()
        all_zu.append(zu)
        all_zs.append(zs)
        all_y.append(y.cpu().numpy())
        all_env.append(env.cpu().numpy())

    if not all_zu:
        raise RuntimeError("No samples were loaded for feature extraction.")

    return (
        np.concatenate(all_zu, axis=0),
        np.concatenate(all_zs, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_env, axis=0),
    )


def sample_points(
    zu: np.ndarray,
    zs: np.ndarray,
    labels: np.ndarray,
    envs: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = zu.shape[0]
    if total <= max_points:
        return zu, zs, labels, envs

    rng = np.random.RandomState(seed)
    unique_labels = sorted(np.unique(labels).tolist())
    if len(unique_labels) <= 1:
        indices = rng.choice(total, size=max_points, replace=False)
        return zu[indices], zs[indices], labels[indices], envs[indices]

    label_to_indices = {
        int(label): np.where(labels == label)[0]
        for label in unique_labels
    }
    for idxs in label_to_indices.values():
        rng.shuffle(idxs)

    target_per_class = max_points // len(unique_labels)
    selected = []
    leftovers = []

    for label in unique_labels:
        idxs = label_to_indices[int(label)]
        take = min(len(idxs), target_per_class)
        if take > 0:
            selected.extend(idxs[:take].tolist())
        if take < len(idxs):
            leftovers.extend(idxs[take:].tolist())

    remaining = max_points - len(selected)
    if remaining > 0 and leftovers:
        leftovers = np.array(leftovers)
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining].tolist())

    if len(selected) < max_points:
        unselected = np.setdiff1d(np.arange(total), np.array(selected), assume_unique=False)
        if len(unselected):
            rng.shuffle(unselected)
            selected.extend(unselected[:(max_points - len(selected))].tolist())

    indices = np.array(selected[:max_points])
    rng.shuffle(indices)
    return zu[indices], zs[indices], labels[indices], envs[indices]


def tsne_project(features: np.ndarray, perplexity: float, iterations: int, seed: int) -> np.ndarray:
    if features.shape[0] < 2:
        raise RuntimeError("Need at least 2 samples to run t-SNE.")
    reduced = features
    if features.shape[1] > 50:
        pca_dim = min(50, features.shape[0], features.shape[1])
        reduced = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)

    effective_perplexity = min(float(perplexity), max(1.0, float(features.shape[0] - 1)))
    if effective_perplexity != float(perplexity):
        print(
            f"Adjusted t-SNE perplexity from {perplexity} to {effective_perplexity} "
            f"to match sample count {features.shape[0]}."
        )

    tsne_kwargs = dict(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    try:
        return TSNE(max_iter=iterations, **tsne_kwargs).fit_transform(reduced)
    except TypeError:
        return TSNE(n_iter=iterations, **tsne_kwargs).fit_transform(reduced)


def infer_env_ids(dataset_obj) -> List[int]:
    try:
        return list(range(len(dataset_obj)))
    except TypeError:
        pass
    except Exception:
        pass

    for attr in ("num_envs", "N_ENVS"):
        value = getattr(dataset_obj, attr, None)
        if value is not None:
            return list(range(int(value)))

    raise RuntimeError("Could not infer environment ids from dataset object.")


def get_num_workers(dataset_obj, args) -> int:
    if hasattr(dataset_obj, "N_WORKERS"):
        return int(getattr(dataset_obj, "N_WORKERS"))
    return int(args.num_workers)


def build_class_style_map(class_ids: Sequence[int]) -> Dict[int, Dict[str, object]]:
    class_ids = sorted(set(int(class_id) for class_id in class_ids))
    cmap_name = "tab10" if len(class_ids) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    marker_bank = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    style_map = {}
    for idx, class_id in enumerate(class_ids):
        style_map[class_id] = {
            "color": cmap(idx % cmap.N),
            "marker": marker_bank[idx % len(marker_bank)],
        }
    return style_map


def scatter_by_class(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    class_style_map: Dict[int, Dict[str, object]],
    title: Optional[str],
) -> None:
    unique_labels = sorted(np.unique(labels).tolist())
    for class_id in unique_labels:
        idx = np.where(labels == class_id)[0]
        if idx.size == 0:
            continue
        style = class_style_map[int(class_id)]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=24,
            alpha=0.8,
            c=[style["color"]],
            marker=style["marker"],
            edgecolors="black",
            linewidths=0.4,
            label=f"Class {int(class_id)}",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def annotate_class_labels(
    ax,
    class_ids: Sequence[int],
    class_style_map: Dict[int, Dict[str, object]],
) -> None:
    y = 0.96
    base_x = 0.76
    for idx, class_id in enumerate(class_ids):
        style = class_style_map[int(class_id)]
        x = base_x + idx * 0.16
        ax.scatter(
            [x],
            [y],
            s=52,
            c=[style["color"]],
            marker=style["marker"],
            edgecolors="black",
            linewidths=0.6,
            transform=ax.transAxes,
            clip_on=False,
            zorder=5,
        )
        ax.text(
            x + 0.03,
            y,
            f"Class {int(class_id)}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="black",
        )


def build_title(args, checkpoint_name: str, n_points: int, suffix: str = "") -> str:
    title = (
        f"{args.dataset} | {args.algorithm} | {checkpoint_name} | N={n_points}"
    )
    if suffix:
        title = f"{title} | {suffix}"
    return title


def ensure_output_dir(source_dir: str, dataset: str) -> str:
    normalized = os.path.normpath(source_dir)
    parts = normalized.split(os.sep)

    sweep_root = None
    if "sweep" in parts:
        sweep_idx = parts.index("sweep")
        if sweep_idx + 1 < len(parts):
            sweep_root = os.sep.join(parts[:sweep_idx + 2])
            if normalized.startswith(os.sep):
                sweep_root = os.sep + sweep_root.lstrip(os.sep)

    if sweep_root is None:
        sweep_root = os.path.join("sweep", dataset)

    out_dir = os.path.join(sweep_root, "tsne")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def ensure_run_output_dir(run_dir: str, dataset: str) -> str:
    out_dir = os.path.join(run_dir, "tsne", dataset)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def project_features_or_stub(
    features: np.ndarray,
    perplexity: float,
    iterations: int,
    seed: int,
    env_id: int,
    feature_name: str,
) -> np.ndarray:
    if features.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if features.shape[0] == 1:
        print(f"Warning: env {env_id} has only 1 sample for {feature_name}; using a zero placeholder instead of t-SNE.")
        return np.zeros((1, 2), dtype=np.float32)
    return tsne_project(features, perplexity, iterations, seed)


def sample_global_points(
    zu: np.ndarray,
    zs: np.ndarray,
    labels: np.ndarray,
    envs: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = zu.shape[0]
    if total <= max_points:
        return zu, zs, labels, envs
    rng = np.random.RandomState(seed)
    indices = rng.choice(total, size=max_points, replace=False)
    return zu[indices], zs[indices], labels[indices], envs[indices]


def balance_by_env_and_class(
    zu: np.ndarray,
    zs: np.ndarray,
    labels: np.ndarray,
    envs: np.ndarray,
    seed: int,
    points_per_group: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    groups = []
    for env_id in sorted(np.unique(envs).tolist()):
        env_mask = (envs == env_id)
        env_labels = labels[env_mask]
        for class_id in sorted(np.unique(env_labels).tolist()):
            group_idx = np.where(env_mask & (labels == class_id))[0]
            if group_idx.size:
                groups.append(group_idx)

    if not groups:
        return zu, zs, labels, envs

    min_group_size = min(group.size for group in groups)
    if points_per_group is not None and points_per_group > 0:
        min_group_size = min(min_group_size, points_per_group)

    if min_group_size <= 0:
        return zu, zs, labels, envs

    selected = []
    for group in groups:
        shuffled = group.copy()
        rng.shuffle(shuffled)
        selected.extend(shuffled[:min_group_size].tolist())

    selected = np.array(selected)
    rng.shuffle(selected)
    return zu[selected], zs[selected], labels[selected], envs[selected]


def plot_mixed_environment_tsne(
    model: torch.nn.Module,
    dataset_obj,
    run_args: Dict[str, object],
    selection: CheckpointSelection,
    device: torch.device,
    args,
) -> None:
    all_env_ids = infer_env_ids(dataset_obj)
    env_ids = args.envs if args.envs else all_env_ids
    env_ids = [int(env_id) for env_id in env_ids]
    invalid_envs = [env_id for env_id in env_ids if env_id not in all_env_ids]
    if invalid_envs:
        raise ValueError(f"Invalid env ids {invalid_envs}. Available env ids: {all_env_ids}")

    num_workers = get_num_workers(dataset_obj, args)
    all_zu, all_zs, all_labels, all_envs = [], [], [], []
    points_per_env = {}

    for env_id in env_ids:
        loader = DataLoader(
            dataset_obj[env_id],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        zu, zs, labels, envs = collect_features(model, loader, device, fallback_env=env_id)
        if np.any(envs < 0):
            envs = np.full_like(labels, env_id)
        if len(labels) == 0:
            continue
        all_zu.append(zu)
        all_zs.append(zs)
        all_labels.append(labels)
        all_envs.append(envs)
        points_per_env[str(env_id)] = int(len(labels))

    if not all_zu:
        raise RuntimeError("No samples were collected for mixed-environment t-SNE.")

    all_zu = np.concatenate(all_zu, axis=0)
    all_zs = np.concatenate(all_zs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_envs = np.concatenate(all_envs, axis=0)

    all_zu, all_zs, all_labels, all_envs = balance_by_env_and_class(
        all_zu, all_zs, all_labels, all_envs, args.seed, args.max_points
    )

    tsne_zu = tsne_project(all_zu, args.tsne_perplexity, args.tsne_iter, args.seed)
    tsne_zs = tsne_project(all_zs, args.tsne_perplexity, args.tsne_iter, args.seed)

    class_ids = sorted(set(int(label) for label in all_labels.tolist()))
    env_ids_present = sorted(set(int(env_id) for env_id in all_envs.tolist()))
    cmap = plt.cm.tab10
    class_colors_left = {
        class_id: cmap(idx % 10)
        for idx, class_id in enumerate(class_ids)
    }
    env_colors_right = {
        env_id: cmap((idx + 4) % 10)
        for idx, env_id in enumerate(env_ids_present)
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix"
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for class_id in class_ids:
        for env_id in env_ids_present:
            idx = np.where((all_labels == class_id) & (all_envs == env_id))[0]
            if idx.size == 0:
                continue
            axes[0].scatter(
                tsne_zu[idx, 0],
                tsne_zu[idx, 1],
                s=22,
                c=[class_colors_left[class_id]],
                marker='o',
                edgecolors="black",
                linewidths=0.35,
                alpha=0.85,
            )

    for env_id in env_ids_present:
        for class_id in class_ids:
            idx = np.where((all_envs == env_id) & (all_labels == class_id))[0]
            if idx.size == 0:
                continue
            axes[1].scatter(
                tsne_zs[idx, 0],
                tsne_zs[idx, 1],
                s=22,
                c=[env_colors_right[env_id]],
                marker='o',
                edgecolors="black",
                linewidths=0.35,
                alpha=0.85,
            )

    axes[0].set_title(r"t-SNE($z_u$) (Invariant Features)", fontsize=14, fontweight="bold")
    axes[1].set_title(r"t-SNE($z_s$) (Variant Features)", fontsize=14, fontweight="bold")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    class_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f"Class {class_id}",
            markerfacecolor=class_colors_left[class_id],
            markeredgecolor='black',
            markersize=7,
            linewidth=0,
        )
        for class_id in class_ids
    ]
    env_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f"Env {env_id}",
            markerfacecolor=env_colors_right[env_id],
            markeredgecolor='black',
            markersize=7,
            linewidth=0,
        )
        for env_id in env_ids_present
    ]

    legend_left = axes[0].legend(
        handles=class_handles,
        loc="best",
        frameon=False,
        fontsize=11,
    )
    legend_right = axes[1].legend(
        handles=env_handles,
        loc="best",
        frameon=False,
        fontsize=11,
    )
    for text in legend_left.get_texts():
        text.set_fontweight("bold")
    for text in legend_right.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout()

    output_dir = ensure_output_dir(args.input_dir or args.model_dir, args.dataset)
    figure_path = os.path.join(output_dir, "tsne_mixed_env.png")
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mixed-environment figure: {figure_path}")


def main(args) -> None:
    set_seed(args.seed)
    device = infer_device(args.device)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })

    search_dir = args.model_dir or args.input_dir
    if search_dir is None:
        raise ValueError("Either --model_dir or --input_dir must be specified.")

    selection = select_checkpoint_from_input_dir(
        search_dir, args.dataset, args.algorithm, args.test_env
    )

    checkpoint_bundle = load_checkpoint_bundle(selection.checkpoint_path)
    model = load_model_from_bundle(checkpoint_bundle)
    model.to(device)
    model.eval()

    run_args = checkpoint_bundle.get("args", {})
    hparams = checkpoint_bundle.get("model_hparams")
    if hparams is None:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)

    dataset_class = datasets.get_dataset_class(args.dataset)
    constructor_test_envs = []
    if getattr(dataset_class, "ENVIRONMENTS", None) is not None:
        constructor_test_envs = list(range(len(dataset_class.ENVIRONMENTS)))
    dataset_obj = dataset_class(args.data_dir, constructor_test_envs, hparams)

    if args.mixed_env:
        plot_mixed_environment_tsne(
            model=model,
            dataset_obj=dataset_obj,
            run_args=run_args,
            selection=selection,
            device=device,
            args=args,
        )
        return

    num_workers = get_num_workers(dataset_obj, args)

    all_env_ids = infer_env_ids(dataset_obj)
    env_ids = args.envs if args.envs else [args.test_env]
    env_ids = [int(env_id) for env_id in env_ids]
    invalid_envs = [env_id for env_id in env_ids if env_id not in all_env_ids]
    if invalid_envs:
        raise ValueError(f"Invalid env ids {invalid_envs}. Available env ids: {all_env_ids}")

    env_results = []
    all_labels_seen = []
    points_per_env = {}

    for env_id in env_ids:
        env_dataset = dataset_obj[env_id]
        loader = DataLoader(
            env_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        zu, zs, labels, envs = collect_features(model, loader, device, fallback_env=env_id)
        if np.any(envs < 0):
            envs = np.full_like(labels, env_id)
        zu, zs, labels, envs = sample_points(
            zu, zs, labels, envs, args.max_points_per_env, args.seed + env_id
        )

        if not len(labels):
            continue

        zu_2d = project_features_or_stub(
            zu, args.tsne_perplexity, args.tsne_iter, args.seed, env_id, "zu"
        )
        zs_2d = project_features_or_stub(
            zs, args.tsne_perplexity, args.tsne_iter, args.seed, env_id, "zs"
        )

        env_results.append(
            {
                "env_id": env_id,
                "labels": labels,
                "zu_2d": zu_2d,
                "zs_2d": zs_2d,
            }
        )
        all_labels_seen.extend(labels.tolist())
        points_per_env[str(env_id)] = int(len(labels))

    if not env_results:
        raise RuntimeError("No labeled samples were collected for visualization.")

    output_root = args.input_dir or args.model_dir
    output_dir = ensure_output_dir(output_root, args.dataset)
    figure_path = os.path.join(output_dir, "tsne_env_rows_zu_zs.png")
    summary_path = os.path.join(output_dir, "summary.json")

    num_classes = getattr(model, "num_classes", None)
    if num_classes is None:
        num_classes = getattr(dataset_obj, "num_classes", None)
    if num_classes is not None:
        class_ids = list(range(int(num_classes)))
    else:
        class_ids = sorted(set(int(label) for label in all_labels_seen))
    class_style_map = build_class_style_map(class_ids)

    rows = len(env_results)
    fig, axes = plt.subplots(rows, 2, figsize=(14, max(4, 4 * rows)), squeeze=False)
    for row_idx, env_result in enumerate(env_results):
        labels = env_result["labels"]
        scatter_by_class(
            axes[row_idx, 0],
            env_result["zu_2d"],
            labels,
            class_style_map,
            None,
        )
        scatter_by_class(
            axes[row_idx, 1],
            env_result["zs_2d"],
            labels,
            class_style_map,
            None,
        )
        annotate_class_labels(axes[row_idx, 0], class_ids, class_style_map)
        annotate_class_labels(axes[row_idx, 1], class_ids, class_style_map)

    for row_idx, env_result in enumerate(env_results):
        fig.text(
            0.025,
            1 - (row_idx + 0.5) / len(env_results),
            f"env={env_result['env_id']}",
            rotation=90,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
        )

    fig.text(
        0.28,
        0.035,
        r"Left: t-SNE($z_u$) (Invariant Features)",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold"
    )
    fig.text(
        0.72,
        0.035,
        r"Right: t-SNE($z_s$) (Variant Features)",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold"
    )

    fig.tight_layout(rect=[0.05, 0.08, 1, 0.98])
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {figure_path}")

    total_points = int(sum(points_per_env.values()))
    summary = {
        "checkpoint_path": selection.checkpoint_path,
        "checkpoint_name": selection.checkpoint_name,
        "selection_reason": selection.selection_reason,
        "metric_name": selection.metric_name,
        "metric_value": selection.metric_value,
        "metric_source": selection.metric_source,
        "dataset": args.dataset,
        "algorithm": args.algorithm,
        "split": "train",
        "device": str(device),
        "seed": args.seed,
        "tsne_perplexity": args.tsne_perplexity,
        "tsne_iter": args.tsne_iter,
        "env_ids": env_ids,
        "points_per_env": points_per_env,
        "total_points": total_points,
        "max_points_per_env": args.max_points_per_env,
        "output_dir": output_dir,
        "figure_path": figure_path,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE visualization of VITA zu/zs features")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--test_env", type=int, default=0)
    parser.add_argument("--envs", type=int, nargs="+", default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mixed_env", action="store_true")
    parser.add_argument("--max_points", type=int, default=4000)
    parser.add_argument("--max_points_per_env", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=1500)
    args = parser.parse_args()
    main(args)
