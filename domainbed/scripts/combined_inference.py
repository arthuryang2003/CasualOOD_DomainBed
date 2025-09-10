
import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from domainbed import datasets, algorithms
from domainbed.lib import reporting
from domainbed import model_selection

# ----------------------------
# Utilities
# ----------------------------

def find_best_model_dir(input_dir: str, dataset: str, algorithm: str, test_env: int) -> str:
    """Return the output directory of the best run for the given setting."""
    records = reporting.load_records(input_dir)
    records = reporting.get_grouped_records(records)
    records = records.filter(
        lambda r: r['dataset'] == dataset and
                  r['algorithm'] == algorithm and
                  r['test_env'] == test_env
    )

    if not len(records):
        raise RuntimeError('No records found for the specified configuration')

    best_dir = None
    best_val = -float('inf')

    for group in records:
        hparams_accs = model_selection.OracleSelectionMethod.hparams_accs(group['records'])
        if not hparams_accs:
            continue
        run_acc, run_records = hparams_accs[0]
        if run_acc['val_acc'] > best_val:
            best_val = run_acc['val_acc']
            best_dir = run_records[0]['args']['output_dir']

    if best_dir is None:
        raise RuntimeError('Could not determine best model directory')

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


@torch.no_grad()
def least_squares_correction(Y_unstable: torch.Tensor,
                             e_matrix: torch.Tensor,
                             iters: int = 300,
                             lr: float = 0.05) -> torch.Tensor:
    """
    迭代式最小二乘校正（多分类）。
    Y_unstable: [B, C]，不稳定分支（tilde_s_logits）的softmax概率
    e_matrix:   [C, C]，由稳定分支概率构造的“映射/混淆”矩阵（经验性估计）
    返回:       [B, C]，校正后的不稳定分支概率
    """
    B, C = Y_unstable.shape
    # 初始化为均匀分布（也可初始化为 Y_unstable）
    p = torch.full_like(Y_unstable, 1.0 / C)

    for _ in range(iters):
        # 目标: 最小化 || E p - y ||_F^2
        Ep = torch.matmul(p, e_matrix.T)                # [B, C]
        grad = torch.matmul(Ep - Y_unstable, e_matrix)  # [B, C]
        p = p - lr * grad
        p = F.softmax(p, dim=1)  # 投影回概率单纯形

    return p

@torch.no_grad()
def stable_acc(model: algorithms.Algorithm,
               loader: torch.utils.data.DataLoader,
               num_classes: int,
               device: torch.device) -> float:
    """Accuracy of u_logits predictions over the given loader."""
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        _, _, u_logits, _, _, _ = model.encode(x)
        if num_classes == 2:
            pred = (torch.sigmoid(u_logits).squeeze(-1) > 0.5).long()
            pred = pred.argmax(dim=1)
        else:
            pred = torch.argmax(u_logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def combined_inference(model: algorithms.Algorithm,
                       loader: torch.utils.data.DataLoader,
                       num_classes: int,
                       device: torch.device) -> float:
    """
    在整个 test 数据集上执行联合推理，返回 accuracy（百分数）。

    要求 model.encode(x) -> (z_u, z_s, u_logits, s_logits, tilde_s_logits, combined_logits)
    其中：
      - u_logits          稳定分支输出
      - tilde_s_logits    不稳定分支校正后的输出（或未校正logits，用于本函数再做校正）
    """
    model.eval()

    if num_classes == 2:
        # ---------- Step 1: 估计 PY, e0, e1 ----------
        PY_sum = 0.0
        n = 0
        n1 = 0.0
        e0_sum = 0.0
        e1_sum = 0.0

        for batch in loader:
            x = batch[0].to(device)
            z_u, z_s, u_logits, s_logits, tilde_s_logits, _ = model.encode(x)
            y_stable = torch.sigmoid(u_logits).squeeze(-1)  # [B]

            PY_sum += y_stable.sum().item()
            n += y_stable.numel()
            n1 += y_stable.sum().item()

            e0_sum += ((1 - y_stable) ** 2).sum().item()
            e1_sum += (y_stable ** 2).sum().item()

        PY = PY_sum / max(n, 1)
        e0 = e0_sum / max((n - n1), 1e-6)
        e1 = e1_sum / max(n1, 1e-6)

        # ---------- Step 2: 校正 + 联合推理 ----------
        correct = 0
        total = 0
        eps = 1e-6
        log_prior = np.log(PY / max(1.0 - PY, eps)) if 0.0 < PY < 1.0 else 0.0

        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            z_u, z_s, u_logits, s_logits, tilde_s_logits, _ = model.encode(x)
            y_stable = torch.sigmoid(u_logits).squeeze(-1)          # [B]
            y_unstable = torch.sigmoid(tilde_s_logits).squeeze(-1)  # [B]

            x_logit = torch.logit(y_stable.clamp(eps, 1 - eps))

            denom = (e1 + e0 - 1.0)
            if abs(denom) < 1e-6:
                y_unstable_corrected = y_unstable
            else:
                y_unstable_corrected = (y_unstable + e0 - 1.0) / denom
                y_unstable_corrected = y_unstable_corrected.clamp(0.0, 1.0)

            u_logit = torch.logit(y_unstable_corrected.clamp(eps, 1 - eps))

            combined_logit = x_logit + u_logit - log_prior
            pred = (torch.sigmoid(combined_logit).squeeze(-1) > 0.5).long()
            pred = pred.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

        return 100.0 * correct / max(total, 1)

    else:
        # ---------- Step 1: 估计 PY 与 e_matrix ----------
        PY_raw = None
        y_soft_all = []

        for batch in loader:
            x = batch[0].to(device)
            z_u, z_s, u_logits, s_logits, tilde_s_logits, _ = model.encode(x)
            stable_pred = F.softmax(u_logits, dim=1)  # [B, C]
            y_soft_all.append(stable_pred)
            if PY_raw is None:
                PY_raw = stable_pred.sum(dim=0)       # [C]
            else:
                PY_raw += stable_pred.sum(dim=0)

        PY = PY_raw / PY_raw.sum()                    # [C]
        PY = PY.clamp(min=1e-6)
        PY = PY / PY.sum()

        Y_all = torch.cat(y_soft_all, dim=0)          # [N, C]
        # 经验映射矩阵（可根据需要更换为你的统计方式）
        e_matrix = torch.matmul(Y_all.T, F.normalize(Y_all, p=1, dim=1))  # [C, C]
        e_matrix = e_matrix / (e_matrix.sum(dim=1, keepdim=True) + 1e-6)

        # ---------- Step 2: 校正 + 联合推理 ----------
        correct = 0
        total = 0
        log_PY = torch.log(PY).to(device)

        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            z_u, z_s, u_logits, s_logits, tilde_s_logits, _ = model.encode(x)
            stable_prob = F.softmax(u_logits, dim=1)         # [B, C]
            unstable_prob = F.softmax(tilde_s_logits, dim=1) # [B, C]

            unstable_corrected = least_squares_correction(unstable_prob, e_matrix.to(device))

            stable_log = torch.log(stable_prob.clamp_min(1e-6))
            unstable_log = torch.log(unstable_corrected.clamp_min(1e-6))
            combined_log = stable_log + unstable_log - log_PY  # [B, C]
            pred = torch.argmax(combined_log, dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

        return 100.0 * correct / max(total, 1)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 选择模型目录
    model_dir = args.model_dir
    if model_dir is None:
        if args.input_dir is None:
            raise ValueError('Either --model_dir or --input_dir must be specified')
        model_dir = find_best_model_dir(args.input_dir, args.dataset, args.algorithm, args.test_env)

    # 加载模型
    model = load_model(os.path.join(model_dir, 'model.pkl'))
    model.to(device)
    model.eval()

    # 构建数据集与 DataLoader（整份 test，评估用，不打乱）
    dataset = datasets.get_dataset_class(args.dataset)(
        args.data_dir, [args.test_env], model.hparams
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset[args.test_env],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=dataset.N_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # 类别数
    num_classes = model.num_classes if hasattr(model, 'num_classes') else dataset.num_classes

    # 评估
    combined_acc = combined_inference(model, eval_loader, num_classes, device)
    stable_only_acc = stable_acc(model, eval_loader, num_classes, device)
    msg = (
        f"[Combined Inference] Test accuracy: {combined_acc:.2f}% "
        f"(u_logits only: {stable_only_acc:.2f}%)"
    )
    print(msg)

    # 可选：保存到文件
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        save_path = os.path.join(args.out_dir, 'combined_inference.txt')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(msg + "\n")
        print(f"Saved: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combined inference on test set')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--test_env', type=int, default=0)
    parser.add_argument('--model_dir', type=str, help='Path to a single run directory containing model.pkl')
    parser.add_argument('--input_dir', type=str, help='Sweep root to auto-pick best run')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out_dir', type=str, default=None, help='Optional directory to save result text')
    args = parser.parse_args()
    main(args)
