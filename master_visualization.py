import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm

# =====================================================================
# Professional Matplotlib Global Configurations
# =====================================================================
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'axes.linewidth': 2.0,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

DISPLAY_NAMES = {
    "ssftt": "SSFTT",
    "mamba": "Mamba",
    "moe_mamba": "MoE-Mamba",
    "ssrn": "SSRN",
    "a2s2kresnet": "A2S2K-ResNet",
    "contextualnet": "ContextualNet",
    "cnn2d": "CNN-2D",
    "cnn3d": "CNN-3D",
    "hybridsn": "HybridSN",
    "morphformer": "MorphFormer"
}

MODEL_COLORS = {
    "ssftt": "#1f77b4",     # Deep blue
    "mamba": "#ff7f0e",     # Vibrant orange
    "moe_mamba": "#2ca02c", # Rich green
    "ssrn": "#9467bd",      # Purple
    "a2s2kresnet": "#8c564b", # Brown
    "contextualnet": "#e377c2", # Pink
    "cnn2d": "#d62728",     # Red
    "cnn3d": "#17becf",     # Cyan
    "hybridsn": "#7f7f7f",  # Gray
    "morphformer": "#bcbd22" # Olive
}

MODEL_MARKERS = {
    "ssftt": "o",
    "mamba": "s",
    "moe_mamba": "^",
    "ssrn": "D",
    "a2s2kresnet": "v",
    "contextualnet": "p",
    "cnn2d": "*",
    "cnn3d": "X",
    "hybridsn": "h",
    "morphformer": "8"
}


# =====================================================================
# Master Classification Map Generator (User Template)
# =====================================================================
def generate_classification_map(
    model,
    dataset,
    device='cuda',
    run_dir=None,
    dataset_name=None,
    model_name=None,
    run_number=None,
    cmap='tab20',
    show_colorbar=False,
    dpi=300,
    block_background=True,
    mode='labeled_only',
    use_spy_colors=True,
    mask_background_in_full_map=False   # ✅ NEW FLAG
):
    """
    Generate pixel-wise classification map for HSI datasets.
    """
    print("\n" + "=" * 60)
    print(f"Generating Classification Map for {model_name} on {dataset_name}...")
    print("=" * 60)

    model = model.to(device)
    model.eval()

    bands, height, width = dataset.data.shape
    gt_height, gt_width = dataset.gt.shape
    half_patch = dataset.patch_size // 2

    print(f"  Data shape: ({bands}, {height}, {width})")
    print(f"  GT shape:   ({gt_height}, {gt_width})")

    prediction_map = np.zeros((gt_height, gt_width), dtype=np.int32)
    map_height = min(height, gt_height)
    map_width = min(width, gt_width)

    if height != gt_height or width != gt_width:
        print(f"  Warning: Data dims ({height},{width}) != GT dims ({gt_height},{gt_width})")
        print(f"  Using overlap region: ({map_height},{map_width})")

    padded_data = np.pad(
        dataset.data,
        ((0, 0), (half_patch, half_patch), (half_patch, half_patch)),
        mode='reflect'
    )

    batch_size = getattr(dataset, 'inference_batch_size', 64)
    batch_patches = []
    batch_positions = []

    total_pixels = map_height * map_width
    pbar = tqdm(total=total_pixels, desc="Classifying pixels", unit="px")

    for i in range(map_height):
        for j in range(map_width):

            patch = padded_data[:, i:i + dataset.patch_size, j:j + dataset.patch_size]

            if dataset.use_channel_dim:
                patch = patch.reshape((1,) + patch.shape)

            batch_patches.append(patch)
            batch_positions.append((i, j))

            is_last = (i == map_height - 1 and j == map_width - 1)

            if len(batch_patches) == batch_size or is_last:
                if len(batch_patches) == 0:
                    continue

                patches_tensor = torch.from_numpy(
                    np.stack(batch_patches)
                ).float().to(device)

                with torch.no_grad():
                    outputs = model(patches_tensor)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    elif isinstance(outputs, dict):
                        outputs = outputs.get('logits', outputs[list(outputs.keys())[0]])
                    preds = outputs.argmax(dim=1).cpu().numpy()

                for idx, (y, x) in enumerate(batch_positions):
                    prediction_map[y, x] = preds[idx] + 1

                pbar.update(len(batch_patches))
                batch_patches.clear()
                batch_positions.clear()

    pbar.close()

    if mode == 'full_image':
        print("  Visualization mode: Full Image")
        vis_prediction = prediction_map.copy()
        if mask_background_in_full_map and hasattr(dataset, 'gt'):
            print("  Applying GT background mask in full map")
            vis_prediction[dataset.gt == 0] = 0

        if use_spy_colors:
            try:
                import spectral as spy
                from spectral import spy_colors
                out_path = os.path.join(run_dir, f'cls_map_{model_name}_{dataset_name}_full.png') if run_dir else f"cls_map_{model_name}_{dataset_name}_full.png"
                spy.save_rgb(out_path, vis_prediction, colors=spy_colors)
                print(f"✓ Full classification map saved: {out_path}")
                if hasattr(dataset, 'gt'):
                    gt_path = os.path.join(run_dir, f'gt_{dataset_name}.png') if run_dir else f"gt_{dataset_name}.png"
                    spy.save_rgb(gt_path, dataset.gt, colors=spy_colors)
                return prediction_map
            except ImportError:
                print("Warning: spectral not installed. Falling back to matplotlib.")
                use_spy_colors = False

        out_path = os.path.join(run_dir, f'cls_map_{model_name}_{dataset_name}_full_{cmap}.png') if run_dir else f"cls_map_{model_name}_{dataset_name}_full_{cmap}.png"
        _save_with_matplotlib(vis_prediction, out_path, cmap, show_colorbar, dpi, block_background=False)

    else:
        print("  Visualization mode: Labeled Regions Only")
        if hasattr(dataset, 'gt'):
            prediction_map[dataset.gt == 0] = 0

        out_path = os.path.join(run_dir, f'cls_map_{model_name}_{dataset_name}_{cmap}.png') if run_dir else f"cls_map_{model_name}_{dataset_name}_{cmap}.png"
        _save_with_matplotlib(prediction_map, out_path, cmap, show_colorbar, dpi, block_background)
        print(f"✓ Classification map saved: {out_path}")

    return prediction_map

def _save_with_matplotlib(prediction_map, output_path, cmap, show_colorbar, dpi, block_background, generic_path=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    current_cmap = plt.get_cmap(cmap).copy()

    if block_background:
        masked = np.ma.masked_where(prediction_map == 0, prediction_map)
        current_cmap.set_bad(color='black')
        im = ax.imshow(masked, cmap=current_cmap, interpolation='nearest')
    else:
        vmax = prediction_map.max()
        im = ax.imshow(prediction_map, cmap=current_cmap, interpolation='nearest', vmin=0, vmax=vmax if vmax > 0 else None)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Class Label', rotation=270, labelpad=20)

    ax.axis('off')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    if generic_path:
        plt.savefig(generic_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


# =====================================================================
# Master PR Curve Generator (Reads from NPZ or TXT)
# =====================================================================
def plot_master_pr_curves(dataset, loss, bit, models, result_dir="output", output_dir="master_visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    valid_plots = 0
    
    for model in models:
        # Check for NPZ file from comprehensive study
        run_name = f"{model}_{dataset}_{loss}_pca30_patch11_bits{bit}"
        npz_path = os.path.join(result_dir, f"{run_name}_pr.npz")
        
        # Check for TXT file from base training
        txt_path = os.path.join(result_dir, f"{dataset}_{loss}_{model}_Bit{bit}.txt")
        
        P, R = None, None
        
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            R, P = data['R'], data['P']
        elif os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            pr_line = next((line for line in reversed(lines) if "PR |" in line), None)
            if pr_line:
                data = [d for d in pr_line[pr_line.rfind("|")+2:-2].strip().split(' ') if d]
                P = [float(data[j]) for j in range(len(data)) if j % 2 == 0]
                R = [float(data[j]) for j in range(len(data)) if j % 2 == 1]
        
        if P is not None and R is not None:
            color = MODEL_COLORS.get(model, "#333333")
            marker = MODEL_MARKERS.get(model, "o")
            display_name = DISPLAY_NAMES.get(model, model.upper())
            
            ax.plot(R, P, linestyle="-", color=color, marker=marker, 
                    label=display_name, linewidth=3.5, markersize=10, 
                    markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
            valid_plots += 1
        else:
            print(f"Warning: No PR data found for {model} on {dataset} ({bit} bits, loss={loss})")

    if valid_plots > 0:
        ax.grid(True, linestyle='--', alpha=0.7, color='#B0BEC5')
        ax.set_xlabel('Recall', fontweight='bold', labelpad=15)
        ax.set_ylabel('Precision', fontweight='bold', labelpad=15)
        ax.set_title(f"Precision-Recall Curve: {dataset.upper()} | {loss.upper()} | {bit} Bits", pad=20, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        legend = ax.legend(loc='lower left', shadow=True, borderpad=1, bbox_to_anchor=(1.05, 0.5))
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('#CFD8DC')
        
        out_pr = os.path.join(output_dir, f"master_pr_{dataset}_{loss}_bits{bit}.pdf")
        plt.tight_layout()
        plt.savefig(out_pr, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"✓ Master PR Curve saved to {out_pr}")
    else:
        plt.close(fig)
        print("No valid data found to plot PR curves.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Visualization Script for HSI Hashing")
    parser.add_argument("--mode", type=str, required=True, choices=["pr_curve", "cls_map"], help="What to visualize")
    parser.add_argument("--models", nargs='+', default=list(DISPLAY_NAMES.keys()), help="List of models to visualize")
    parser.add_argument("--datasets", nargs='+', default=["houston2013", "trento", "houston2018", "nilifossae"], help="Datasets")
    parser.add_argument("--losses", nargs='+', default=["csq"], help="Losses to plot PR curves for")
    parser.add_argument("--bits", nargs='+', type=int, default=[64], help="Hash bit lengths to plot")
    parser.add_argument("--result_dir", type=str, default="output", help="Directory where metrics are saved")
    parser.add_argument("--output_dir", type=str, default="master_visualizations", help="Where to save visual outputs")
    
    # Classification map specific arguments
    parser.add_argument("--block_background", action='store_true', help="Set background to black for cls maps")
    parser.add_argument("--mask_background", action='store_true', help="Force GT background pixels to 0 in full map")
    
    args = parser.parse_args()
    
    if args.mode == "pr_curve":
        for dataset in args.datasets:
            for loss in args.losses:
                for bit in args.bits:
                    print(f"Generating PR curves for {dataset}, {loss}, {bit} bits...")
                    plot_master_pr_curves(dataset, loss, bit, args.models, args.result_dir, args.output_dir)
                    
    elif args.mode == "cls_map":
        print("To generate classification maps, this script requires trained model weights and dataset loading.")
        print("You can import `generate_classification_map` from this file in your inference scripts.")
        print("Example usage:")
        print("from master_visualization import generate_classification_map")
        print("generate_classification_map(model=my_model, dataset=my_dataset, run_dir='master_visualizations', model_name='ssftt', dataset_name='houston2013', block_background=True)")
