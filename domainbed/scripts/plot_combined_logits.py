import argparse
import os
from typing import List
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from domainbed import datasets, algorithms
from domainbed.lib import reporting
from domainbed import model_selection
from domainbed.lib.fast_data_loader import VisualizeDataLoader, FastDataLoader


def find_best_model_dir(input_dir: str, dataset: str, algorithm: str, test_env: int) -> str:
    """Return the output directory of the best run for the given setting.

    The search mimics the logic in ``collect_results.py`` by first selecting
    the best run for each hyperparameter seed within every trial (using
    ``OracleSelectionMethod``), then averaging validation accuracy across
    trials. The run with the highest mean validation accuracy determines the
    returned model directory.
    """

    records = reporting.load_records(input_dir)
    grouped = reporting.get_grouped_records(records)
    grouped = grouped.filter(
        lambda r: r['dataset'] == dataset and
        r['algorithm'] == algorithm and
        r['test_env'] == test_env
    )

    if not len(grouped):
        raise RuntimeError('No records found for the specified configuration')

    # Accumulate val accuracies and corresponding dirs per hparams seed
    by_hparams = defaultdict(list)
    for group in grouped:
        hparams_accs = model_selection.OracleSelectionMethod.hparams_accs(group['records'])
        for run_acc, run_records in hparams_accs:
            hseed = run_records[0]['args']['hparams_seed']
            out_dir = run_records[0]['args']['output_dir']
            by_hparams[hseed].append((run_acc['val_acc'], out_dir))

    if not by_hparams:
        raise RuntimeError('Could not determine best model directory')

    # Choose hyperparameter seed with highest mean validation accuracy
    best_seed = None
    best_val = -float('inf')
    for hseed, vals_dirs in by_hparams.items():
        mean_val = np.mean([v for v, _ in vals_dirs])
        if mean_val > best_val:
            best_val = mean_val
            best_seed = hseed

    # Within the best hyperparameter seed, pick the run with highest val_acc
    best_dir = max(by_hparams[best_seed], key=lambda x: x[0])[1]
    return best_dir


def load_model(model_pkl: str) -> algorithms.Algorithm:
    """Load your algorithm instance from model.pkl."""
    checkpoint = torch.load(model_pkl, map_location="cpu")
    alg_name = checkpoint["args"]["algorithm"]
    alg_class = algorithms.get_algorithm_class(alg_name)
    model = alg_class(
        checkpoint["model_input_shape"],
        checkpoint["model_num_classes"],
        checkpoint["model_num_domains"],
        checkpoint["model_hparams"]
    )
    model.load_state_dict(checkpoint["model_dict"])
    return model


def _to_label_index(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert one-hot labels to indices if necessary."""
    if y.ndim == 2 and y.size(1) == num_classes:
        return y.argmax(dim=1)
    return y.view(-1)


@torch.no_grad()
def evaluate_k(model: algorithms.Algorithm,
               loader: torch.utils.data.DataLoader,
               k: float,
               num_classes: int,
               device: torch.device) -> float:
    """Compute accuracy for u_logits + k * tilde_s_logits."""
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        z_u, z_s, u_logits, s_logits, tilde_s_logits, _ = model.encode(x)
        logits = u_logits + k * tilde_s_logits
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


def main(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = args.model_dir
    if model_dir is None:
        if args.input_dir is None:
            raise ValueError('Either --model_dir or --input_dir must be specified')
        model_dir = find_best_model_dir(args.input_dir, args.dataset, args.algorithm, args.test_env)

    model = load_model(os.path.join(model_dir, 'model.pkl'))
    model.to(device)
    model.eval()

    dataset = datasets.get_dataset_class(args.dataset)(
        args.data_dir, [args.test_env], model.hparams
    )
    eval_loader = FastDataLoader(
        dataset=dataset[args.test_env],
        batch_size=args.batch_size,
        num_workers=dataset.N_WORKERS,
    )


    num_classes = model.num_classes if hasattr(model, 'num_classes') else dataset.num_classes
    ks = np.linspace(0.0, 1.0, 11)
    accs: List[float] = []
    for k in ks:
        acc = evaluate_k(model, eval_loader, float(k), num_classes, device)
        accs.append(acc)
        print(f"k={k:.1f}: {acc:.2f}%")

    plt.figure()
    plt.plot(ks, accs, marker='o')
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.title('combined_logits = u_logits + k * tilde_s_logits')
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        fig_path = os.path.join(args.out_dir, 'k_accuracy.png')
        plt.savefig(fig_path)
        data_path = os.path.join(args.out_dir, 'k_accuracy.txt')
        np.savetxt(data_path, np.column_stack([ks, accs]), fmt='%.3f', header='k accuracy')
        print(f'Saved plot to {fig_path}')
        print(f'Saved data to {data_path}')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot accuracy for u_logits + k * tilde_s_logits')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--test_env', type=int, default=0)
    parser.add_argument('--model_dir', type=str, help='Path to a single run directory containing model.pkl')
    parser.add_argument('--input_dir', type=str, help='Sweep root to auto-pick best run')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out_dir', type=str, default=None, help='Directory to save plot and data')
    args = parser.parse_args()
    main(args)
