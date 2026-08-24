import os
import argparse
import matplotlib.pyplot as plt
from utils.tools import draw_range

def plot_curves(dataset, method, bit, models, save_dir="Checkpoints_Results"):
    # --- High-End Academic Plot Aesthetics ---
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'axes.labelsize': 22,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'axes.linewidth': 2.0,
        'grid.alpha': 0.5,
        'grid.linestyle': '--'
    })
    
    # Elegant vibrant colors and distinct markers
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    
    model2color = {model: colors[i % len(colors)] for i, model in enumerate(models)}
    model2marker = {model: markers[i % len(markers)] for i, model in enumerate(models)}
    
    # Beautiful display names for the legend
    DISPLAY_NAMES = {
        "ssftt": "SSFTT",
        "mamba": "Mamba",
        "moe_mamba": "MoE-Mamba"
    }
    
    # Store parsed P and R data for each model
    pr_data = {}
    
    for model in models:
        # Construct path according to our codebase format:
        # e.g., Checkpoints_Results/cifar10_DSHcls_ViT-B_32_Bit64.txt
        file_path = os.path.join(save_dir, f"{dataset}_{method}_{model}_Bit{bit}.txt")
        print(f"Reading {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Find the last PR line (the most recent one saved)
        pr_line = None
        for line in lines:
            if "PR |" in line:
                pr_line = line
                
        if pr_line:
            data = pr_line[pr_line.rfind("|")+2:-2].strip().split(' ')
            data = [d for d in data if d] # remove empty elements
            
            # P is at even indices, R is at odd indices
            P = [float(data[j]) for j in range(len(data)) if j % 2 == 0]
            R = [float(data[j]) for j in range(len(data)) if j % 2 == 1]
            pr_data[model] = (P, R)
        else:
            print(f"Warning: No PR data found in {file_path}")

    if not pr_data:
        print("No data found to plot. Exiting.")
        return

    # 1. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(10, 8))
    for model in pr_data:
        P, R = pr_data[model]
        display_name = DISPLAY_NAMES.get(model, model.upper())
        ax.plot(R, P, linestyle="-", color=model2color[model], marker=model2marker[model], 
                label=display_name, linewidth=3.5, markersize=12, markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.6, color='#B0BEC5')
    ax.set_xlabel('Recall', fontweight='bold', labelpad=15)
    ax.set_ylabel('Precision', fontweight='bold', labelpad=15)
    
    # Remove top and right spines for a clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Enhance legend
    legend = ax.legend(loc='lower left', shadow=True, borderpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#CFD8DC')
    
    out_pr = f"{dataset}_{method}_Bit{bit}_pr.pdf"
    plt.tight_layout()
    plt.savefig(out_pr, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved Precision-Recall Curve to {out_pr}")
    
    # Check if draw_range matches the data length
    # It might vary slightly depending on topK, but usually matches draw_range from utils.tools
    first_model = list(pr_data.keys())[0]
    data_len = len(pr_data[first_model][0])
    x_range = draw_range[:data_len]
    
    # 2. Recall vs Number of Retrieved Samples
    fig, ax = plt.subplots(figsize=(10, 8))
    for model in pr_data:
        P, R = pr_data[model]
        display_name = DISPLAY_NAMES.get(model, model.upper())
        ax.plot(x_range, R, linestyle="-", color=model2color[model], marker=model2marker[model], 
                label=display_name, linewidth=3.5, markersize=12, markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.6, color='#B0BEC5')
    ax.set_xlabel('Number of retrieved samples', fontweight='bold', labelpad=15)
    ax.set_ylabel('Recall', fontweight='bold', labelpad=15)
    ax.set_xlim(0, max(x_range) if x_range else 1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend = ax.legend(loc='lower right', shadow=True, borderpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#CFD8DC')
    
    out_recall = f"{dataset}_{method}_Bit{bit}_recall.pdf"
    plt.tight_layout()
    plt.savefig(out_recall, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved Recall Curve to {out_recall}")
    
    # 3. Precision vs Number of Retrieved Samples
    fig, ax = plt.subplots(figsize=(10, 8))
    for model in pr_data:
        P, R = pr_data[model]
        display_name = DISPLAY_NAMES.get(model, model.upper())
        ax.plot(x_range, P, linestyle="-", color=model2color[model], marker=model2marker[model], 
                label=display_name, linewidth=3.5, markersize=12, markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.6, color='#B0BEC5')
    ax.set_xlabel('Number of retrieved samples', fontweight='bold', labelpad=15)
    ax.set_ylabel('Precision', fontweight='bold', labelpad=15)
    ax.set_xlim(0, max(x_range) if x_range else 1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend = ax.legend(loc='lower right', shadow=True, borderpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#CFD8DC')
    
    out_precision = f"{dataset}_{method}_Bit{bit}_precision.pdf"
    plt.tight_layout()
    plt.savefig(out_precision, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved Precision Curve to {out_precision}")
    
    # Uncomment below if you want interactive plots
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot PR Curves")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--method", type=str, default="DSHcls", help="Method/Info name")
    parser.add_argument("--bit", type=int, default=64, help="Hash bit length")
    parser.add_argument("--models", nargs='+', default=["AlexNet", "ResNet", "ViT-B_32", "ViT-B_16"], help="Models to plot")
    parser.add_argument("--save_dir", type=str, default="Checkpoints_Results", help="Directory containing the log txt files")
    
    args = parser.parse_args()
    
    plot_curves(args.dataset, args.method, args.bit, args.models, args.save_dir)
