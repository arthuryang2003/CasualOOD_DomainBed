import argparse
import os
from typing import Optional, List, Dict

import torch
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt

from domainbed import datasets, algorithms
from domainbed.lib.fast_data_loader import VisualizeDataLoader
from domainbed.lib import reporting
from domainbed import model_selection
from PIL import ImageChops

# -------------------- Utils --------------------
def find_last_conv(module: torch.nn.Module) -> torch.nn.Module:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise ValueError("No Conv2d layer found in model")


def cam_to_pil(cam_tensor: torch.Tensor) -> 'PIL.Image.Image':
    if cam_tensor.ndim == 2:
        cam_tensor = cam_tensor.unsqueeze(0)
    return to_pil_image(cam_tensor, mode='F')


def auto_color_map(img_tensor: torch.Tensor) -> torch.Tensor:
    # img_tensor: CxHxW, normalized
    if img_tensor.shape[0] == 3:
        return img_tensor
    if img_tensor.shape[0] == 2:
        r, g = img_tensor[0:1], img_tensor[1:2]
        b = torch.zeros_like(r)
        return torch.cat([r, g, b], dim=0)
    if img_tensor.shape[0] == 1:
        return img_tensor.repeat(3, 1, 1)
    raise ValueError(f"Unsupported image shape: {img_tensor.shape}")


def find_best_model_dir(input_dir: str, dataset: str, algorithm: str, test_env: int) -> str:
    records = reporting.load_records(input_dir)
    records = reporting.get_grouped_records(records)
    records = records.filter(
        lambda r: r['dataset'] == dataset and r['algorithm'] == algorithm and r['test_env'] == test_env
    )
    if not len(records):
        raise RuntimeError(f'No records found for: dataset={dataset}, alg={algorithm}, test_env={test_env}')

    best_dir, best_val = None, -float('inf')
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

# 小工具：把 CAM 统一成 [H,W] 的 torch.Tensor(float)
def cam_to_2d_tensor(cam_map):
    # 常见：GradCAM 可能返回 [H,W] / [1,H,W] / list(np/torch)
    if isinstance(cam_map, (list, tuple)):
        cam_map = cam_map[0]
    import numpy as np
    if isinstance(cam_map, np.ndarray):
        cam_map = torch.from_numpy(cam_map)
    cam_map = cam_map.squeeze()
    return cam_map.float()

# 简单的 0-1 归一化
def norm01(t):
    tmin, tmax = t.min(), t.max()
    return (t - tmin) / (tmax - tmin + 1e-12)

def select_images(dset, env_idx: int, k: int = 10):
    """从测试域选择 k 张样本，返回 [(CHW tensor, label int), ...]（不做 batch）"""
    # VisualizeDataLoader 里已经封装好 transforms
    dl = VisualizeDataLoader(dset[env_idx], batch_size=1, num_workers=dset.N_WORKERS)
    selected = []
    for x, y in dl:
        selected.append((x[0], int(y[0])))
        if len(selected) >= k:
            break
    return selected

def visualize_for_algorithms(
    data_dir: str,
    dataset_name: str,
    test_env: int,
    algorithms_to_load: List[str],
    model_dirs_override: Dict[str, str],
    input_dir: Optional[str],
    out_root: str,
    k_images: int = 5
):
    """
    生成一张大图（5×5 网格）：
    行 = 样本，列 = [原图] + 各算法GradCAM（ERM/IRM/MMD/VITA）
    仅最上方一行显示大号标题
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 先获取一个参考模型的 hparams
    probe_alg = algorithms_to_load[0]
    if probe_alg in model_dirs_override and model_dirs_override[probe_alg]:
        probe_root = model_dirs_override[probe_alg]
        try:
            probe_model_dir = find_best_model_dir(probe_root, dataset_name, probe_alg, test_env)
        except Exception:
            probe_model_dir = probe_root
    else:
        if input_dir is None:
            raise ValueError("When not providing per-alg model_dir, --input_dir is required.")
        probe_model_dir = find_best_model_dir(input_dir, dataset_name, probe_alg, test_env)

    probe_model = load_model(os.path.join(probe_model_dir, 'model.pkl'))
    probe_model.to(device).eval()

    # 2. 构造数据集并取样
    dset = datasets.get_dataset_class(dataset_name)(data_dir, [test_env], probe_model.hparams)
    samples = select_images(dset, test_env, k=k_images)
    if len(samples) == 0:
        raise RuntimeError("No samples selected from test env.")

    # keep_pos_1based = [4, 5, 6, 9, 10]
    # max_pos = max(keep_pos_1based)
    #
    # # 确保至少抽到 max_pos 张样本（比如 10 张）
    # k_images = max(k_images, max_pos)
    #
    # samples = select_images(dset, test_env, k=k_images)
    # if len(samples) < max_pos:
    #     raise RuntimeError(
    #         f"需要至少 {max_pos} 张样本，但实际只选到了 {len(samples)}。"
    #         f"请增大 k_images 或让 select_images 保证返回数量。"
    #     )
    #
    # # 二次筛选：只保留第 4、5、6、9、10（按 1-based）
    # keep_idx_0based = [p - 1 for p in keep_pos_1based]
    # samples = [samples[i] for i in keep_idx_0based]

    # 3. 加载各算法模型（VITA 用 sweep 方式找最佳）
    alg_models, cam_extractors = {}, {}
    for alg in algorithms_to_load:
        if alg in model_dirs_override and model_dirs_override[alg]:
            root = model_dirs_override[alg]
            try:
                mdir = find_best_model_dir(root, dataset_name, alg, test_env)
                print(f"Found best model for {alg} at {mdir}")
            except Exception:
                mdir = root
        else:
            if input_dir is None:
                raise ValueError(f"Algorithm {alg} needs model dir or --input_dir.")
            mdir = find_best_model_dir(input_dir, dataset_name, alg, test_env)
            print(f"Found best model for {alg} at {mdir}")

        model = load_model(os.path.join(mdir, 'model.pkl'))
        model.to(device).eval()
        alg_models[alg] = model

        target_layer = find_last_conv(model.featurizer)
        cam_extractors[alg] = GradCAM(model, target_layer=target_layer)

    # 4. 构建总图：行=样本，列=Image+算法
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    col_titles = ["Image"] + algorithms_to_load
    ncols = len(col_titles)
    nrows = len(samples)

    fig = plt.figure(figsize=(4*ncols, 4*nrows))
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.02, hspace=0.02)

    for r, (img_chw, label) in enumerate(samples):
        img_vis = auto_color_map(((img_chw * std) + mean).clamp(0,1))
        img_batched = img_chw.unsqueeze(0).to(device)
        img_pil = to_pil_image(img_vis.cpu())

        # 原图列
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(to_pil_image(img_vis.cpu()))
        if r == 0:
            ax.set_title(col_titles[0], fontsize=20, fontweight='bold')
        ax.axis('off')

        # 各算法GradCAM列
        for c, alg in enumerate(algorithms_to_load, start=1):
            model = alg_models[alg]
            cam = cam_extractors[alg]

            img_batched.requires_grad_(True)

            # if alg == 'VITA':
            #     with torch.enable_grad():
            #
            #         z_u, z_s, tilde_z_s, u_logits, s_logits, tilde_s_logits, combined_logits = model.encode(img_batched)
            #         # concat_z = torch.cat([z_u, tilde_z_s], dim=1)
            #         # weights = model.reweighting(concat_z)
            #         class_idx_u = int(u_logits.argmax(dim=1))
            #         class_idx_s = int(tilde_s_logits.argmax(dim=1))
            #         cam_u = cam(class_idx_u, scores=u_logits, retain_graph=True)
            #         cam_s = cam(class_idx_s, scores=tilde_s_logits, retain_graph=True)
            #
            #     # # 取到 [H,W]
            #     # cam_u = cam_to_2d_tensor(cam_u)
            #     # cam_s = cam_to_2d_tensor(cam_s)
            #     #
            #     # # 各自归一化
            #     # cam_u_n = norm01(cam_u)
            #     # cam_s_n = norm01(cam_s)
            #     # cam_mix =cam_u_n+cam_s_n
            #
            #     # 叠加方式：逐像素取最大值（谁亮取谁）
            #     heatmap_u = overlay_mask(img_pil, cam_to_pil(cam_u[0]), alpha=0.5) # -> PIL.Image
            #     heatmap_s = overlay_mask(img_pil, cam_to_pil(cam_s[0]), alpha=0.5)
            #
            #     heat = ImageChops.lighter(heatmap_u, heatmap_s)
            #
            #     # # 一次性着色 + 与原图混合（不再叠彩色 PIL）
            #     # heat = overlay_mask(img_pil, cam_to_pil(cam_mix[0].cpu()), alpha=0.5)
            #
            # else:
            with torch.enable_grad():
                logits = model.predict(img_batched)
                pred_cls = int(logits.argmax(1).item())
                cam_map = cam(pred_cls, scores=logits)

            heat = overlay_mask(img_pil, cam_to_pil(cam_map[0].cpu()), alpha=0.5)

            ax = fig.add_subplot(gs[r, c])
            ax.imshow(heat)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=20, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()

    out_dir = os.path.join(out_root, dataset_name, f"env{test_env}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"biggrid_{nrows}x{ncols}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Big grid saved to: {out_path}")

def parse_alg_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='GradCAM comparison for multiple algorithms')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--input_dir', type=str, help='Root directory containing DomainBed runs')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--test_env', type=int, default=0)
    parser.add_argument('--algorithms', type=str, default='ERM,IRM,MMD,VITA',
                        help='Comma separated list, e.g. ERM,IRM,MMD,VITA')
    parser.add_argument('--out_dir', type=str, default='visualize_cam')
    parser.add_argument('--num_images', type=int, default=10)

    # 可选：为每个算法手动指定模型目录，示例： --model_dir_ERM path1 --model_dir_IRM path2 ...
    parser.add_argument('--model_dir_ERM', type=str, default=None)
    parser.add_argument('--model_dir_IRM', type=str, default=None)
    parser.add_argument('--model_dir_MMD', type=str, default=None)
    parser.add_argument('--model_dir_VITA', type=str, default=None)

    args = parser.parse_args()

    algs = parse_alg_list(args.algorithms)
    override = {
        'ERM': args.model_dir_ERM,
        'IRM': args.model_dir_IRM,
        'MMD': args.model_dir_MMD,
        'VITA': args.model_dir_VITA
    }

    visualize_for_algorithms(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        test_env=args.test_env,
        algorithms_to_load=algs,
        model_dirs_override=override,
        input_dir=args.input_dir,
        out_root=args.out_dir,
        k_images=args.num_images
    )


if __name__ == '__main__':
    main()
