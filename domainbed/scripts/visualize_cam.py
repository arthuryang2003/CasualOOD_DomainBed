import argparse
import os
from typing import Optional

import torch
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt

from domainbed import datasets, algorithms
from domainbed.lib.fast_data_loader import FastDataLoader, VisualizeDataLoader
from domainbed.lib import reporting
from domainbed import model_selection


def find_last_conv(module: torch.nn.Module) -> torch.nn.Module:
    """Return the last Conv2d layer in a module."""
    for layer in reversed(list(module.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise ValueError("No Conv2d layer found in model")


def cam_to_pil(cam_tensor: torch.Tensor) -> 'PIL.Image.Image':
    if cam_tensor.ndim == 2:
        cam_tensor = cam_tensor.unsqueeze(0)
    return to_pil_image(cam_tensor, mode='F')


def auto_color_map(img_tensor: torch.Tensor) -> torch.Tensor:
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
    """Return the output directory of the best run for the given setting."""

    records = reporting.load_records(args.input_dir)
    print("Total records:", len(records))

    records = reporting.get_grouped_records(records)
    records = records.filter(
        lambda r:
            r['dataset'] == args.dataset and
            r['algorithm'] == args.algorithm and
            r['test_env'] == args.test_env
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

def load_model(model_pkl):
    checkpoint = torch.load(model_pkl, map_location="cpu")
    alg_name = checkpoint["args"]["algorithm"]
    alg_class = algorithms.get_algorithm_class(alg_name)
    model = alg_class(
        checkpoint["model_input_shape"],
        checkpoint["model_num_classes"],
        checkpoint["model_num_domains"],
        checkpoint["model_hparams"]   # 包含 nonlinear_classifier
    )
    model.load_state_dict(checkpoint["model_dict"])
    return model

# def load_model(checkpoint_path: str) -> algorithms.Algorithm:
#     ckpt = torch.load(checkpoint_path, map_location='cpu')
#     alg_class = algorithms.get_algorithm_class(ckpt['args']['algorithm'])
#     model = alg_class(
#         ckpt['model_input_shape'],
#         ckpt['model_num_classes'],
#         ckpt['model_num_domains'],
#         ckpt['model_hparams'],
#     )
#     model.load_state_dict(ckpt['model_dict'])
#     return model

def check_constant_predictions(model: algorithms.Algorithm,
                               data_loader: torch.utils.data.DataLoader,
                               device: str) -> Optional[int]:
    """Return 0 or 1 if the model predicts the same class for all samples.

    Parameters
    ----------
    model : algorithms.Algorithm
        Loaded model to evaluate.
    data_loader : DataLoader
        Data loader containing samples to evaluate on.
    device : str
        Device on which computation is performed.

    Returns
    -------
    int | None
        ``0`` if all predictions are class ``0``, ``1`` if all predictions are
        class ``1`` and ``None`` otherwise.
    """

    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            logits = model.predict(x)
            preds.append(logits.argmax(1).cpu())

    if not preds:
        return None

    preds = torch.cat(preds)
    unique = torch.unique(preds)
    if len(unique) == 1 and unique.item() in (0, 1):
        return int(unique.item())
    return None

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dir = args.model_dir
    if model_dir is None:
        if args.input_dir is None:
            raise ValueError('Either --model_dir or --input_dir must be specified')
        model_dir = find_best_model_dir(args.input_dir, args.dataset, args.algorithm, args.test_env)

    model = load_model(os.path.join(model_dir, 'model.pkl'))
    model.to(device)
    model.eval()

    dataset = datasets.get_dataset_class(args.dataset)(
        args.data_dir, [args.test_env], model.hparams)
    test_loader = VisualizeDataLoader(dataset[args.test_env], batch_size=1,
                                 num_workers=dataset.N_WORKERS)
    constant_class = check_constant_predictions(model, test_loader, device)
    if constant_class is not None:
        print(f"Model predicts only class {constant_class} on the test set.")

    out_dir = os.path.join('visualize', args.algorithm, args.dataset)
    os.makedirs(out_dir, exist_ok=True)


    selected = []
    max_images = 20

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        for img, lbl in zip(x, y):
            selected.append((img.unsqueeze(0), lbl.item()))
            if len(selected) >= max_images:
                break
        if len(selected) >= max_images:
            break

    target_layer = find_last_conv(model.featurizer)
    cam_extractor = GradCAM(model, target_layer=target_layer)

    for idx, (img, label_class) in enumerate(selected):
        img = img.to(device)
        img.requires_grad_()

        if args.algorithm == "CasualOODAlgorithm" or args.algorithm == "VITA" :
            with torch.enable_grad():
                z_u, z_s, u_logits, s_logits, tilde_s_logits, combined_logits = model.encode(img)
                class_idx_u = u_logits.argmax(dim=1).item()
                class_idx_s = s_logits.argmax(dim=1).item()
                class_idx_t = tilde_s_logits.argmax(dim=1).item()
                class_idx_c = combined_logits.argmax(dim=1).item()

                cam_map_u = cam_extractor(class_idx_u, scores=u_logits, retain_graph=True)
                cam_map_s = cam_extractor(class_idx_s, scores=s_logits, retain_graph=True)
                cam_map_tilde_s = cam_extractor(class_idx_t, scores=tilde_s_logits, retain_graph=True)
                cam_map_combined = cam_extractor(class_idx_c, scores=combined_logits, retain_graph=True)

            img_vis = auto_color_map(img[0].detach().cpu() * 0.229 + 0.485)
            img_vis = torch.clamp(img_vis, 0, 1)
            heatmap_u = overlay_mask(to_pil_image(img_vis), cam_to_pil(cam_map_u[0]), alpha=0.5)
            heatmap_s = overlay_mask(to_pil_image(img_vis), cam_to_pil(cam_map_s[0]), alpha=0.5)
            heatmap_tilde_s = overlay_mask(to_pil_image(img_vis), cam_to_pil(cam_map_tilde_s[0]), alpha=0.5)
            heatmap_combined = overlay_mask(to_pil_image(img_vis), cam_to_pil(cam_map_combined[0]), alpha=0.5)

            fig, axs = plt.subplots(1, 5, figsize=(16, 4))
            axs[0].imshow(to_pil_image(img_vis))
            axs[0].set_title(f"Original({label_class})")
            axs[1].imshow(heatmap_u)
            axs[1].set_title(f"z_u({class_idx_u})")
            axs[2].imshow(heatmap_s)
            axs[2].set_title(f"z_s({class_idx_s})")
            axs[3].imshow(heatmap_tilde_s)
            axs[3].set_title(f"z_s'({class_idx_t})")
            axs[4].imshow(heatmap_combined)
            axs[4].set_title(f"combined({class_idx_c})")

            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            save_path = os.path.join(out_dir, f"img{idx}_class{label_class}.png")
            plt.savefig(save_path)
            plt.close()
        else:
            with torch.enable_grad():
                logits = model.predict(img)
                class_idx = logits.argmax(1).item()
                cam_map = cam_extractor(class_idx, scores=logits)

            img_vis = auto_color_map(img[0].detach().cpu() * 0.229 + 0.485)
            img_vis = torch.clamp(img_vis, 0, 1)
            heatmap = overlay_mask(to_pil_image(img_vis), cam_to_pil(cam_map[0]), alpha=0.5)

            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            axs[0].imshow(to_pil_image(img_vis))
            axs[0].set_title(f"Original({label_class})")
            axs[1].imshow(heatmap)
            axs[1].set_title(f"{args.algorithm}({class_idx})")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            save_path = os.path.join(out_dir, f"img{idx}_class{label_class}.png")
            plt.savefig(save_path)
            plt.close()

    if hasattr(model, 'mask'):
        mask_sigmoid = torch.sigmoid(model.mask).detach().cpu().numpy().squeeze()
        plt.figure(figsize=(12, 3))
        plt.bar(range(len(mask_sigmoid)), mask_sigmoid)
        plt.title("Mask Channel Weights after Sigmoid")
        plt.xlabel("Channel index")
        plt.ylabel("Gate value (sigmoid)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mask_weights.png"))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GradCAM visualization')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--test_env', type=int, default=0)
    args = parser.parse_args()
    main(args)