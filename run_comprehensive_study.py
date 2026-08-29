import os
import subprocess
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive Ablation Study Orchestrator")
    parser.add_argument("--datasets", nargs='+', default=["houston2013", "trento", "houston2018", "nilifossae"])
    parser.add_argument("--models", nargs='+', default=["ssftt", "mamba", "moe_mamba", "ssrn", "a2s2kresnet", "contextualnet", "cnn2d", "cnn3d", "hybridsn", "morphformer", "spectralformer"])
    parser.add_argument("--losses", nargs='+', default=["csq", "dpn", "dsh", "greedyhash", "hashnet", "idhn", "orthohash", "dspch", "dhnn"])
    parser.add_argument("--bits", nargs='+', type=int, default=[16, 32, 64])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patch", type=int, default=11)
    parser.add_argument("--pca", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="output")
    return parser.parse_args()

def run_experiment(dataset, model, loss, bit, epochs, patch, pca, output_dir):
    # Known prepatched directories. Add more if your structure changes.
    prepatched_map_options = {
        "houston2013": ["../houston13", "cls_SSFTT_IP/houston13_alreadypatched_dataset"],
        "houston2018": ["../Houston18"],
        "trento": ["../Trento"],
        "indian_pines": ["cls_SSFTT_IP/ip_alreadypatched_dataset"],
        "nilifossae": ["../NiliFossae", "cls_SSFTT_IP/NiliFossae_dataset"],
        "longkou": ["WHU-HI_HSI_Dataset_Disjoint_10Train_90Test/WHU-Hi-LongKou", "../WHU-HI_HSI_Dataset_Disjoint_10Train_90Test/WHU-Hi-LongKou", "../WHU-Hi-LongKou"],
        "hanchuan": ["WHU-HI_HSI_Dataset_Disjoint_10Train_90Test/WHU-Hi-HanChuan", "../WHU-HI_HSI_Dataset_Disjoint_10Train_90Test/WHU-Hi-HanChuan", "../WHU-Hi-HanChuan"],
        "honghu": ["WHU-HI_HSI_Dataset_Disjoint_10Train_90Test/WHU-Hi-HongHu", "../WHU-HI_HSI_Dataset_Disjoint_10Train_90Test/WHU-Hi-HongHu", "../WHU-Hi-HongHu"]
    }

    prepatched_map = {}
    for ds_name, paths in prepatched_map_options.items():
        for p in paths:
            if os.path.exists(p):
                prepatched_map[ds_name] = p
                break
        # Fallback to the first path if none exist, so the script can still error out normally
        if ds_name not in prepatched_map:
            prepatched_map[ds_name] = paths[0]
    
    cmd = [
        "python3", "cls_SSFTT_IP/train_hsi_hashing.py",
        "--dataset", dataset,
        "--model", model,
        "--loss_type", loss,
        "--hash_bit_length", str(bit),
        "--epochs", str(epochs),
        "--patch", str(patch),
        "--pca", str(pca),
        "--output_dir", output_dir
    ]
    
    if dataset in prepatched_map:
        cmd.extend(["--prepatched_dir", prepatched_map[dataset]])
        
    print(f"\n[{'='*50}]")
    print(f"Running: {' '.join(cmd)}")
    print(f"[{'='*50}]\n")
    
    # Run the command with PYTHONPATH set
    env = os.environ.copy()
    env["PYTHONPATH"] = "cls_SSFTT_IP"
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred running {model} {loss} {bit}bits for {dataset}. Skipping metrics extraction for this run.")
        pass
    
    # Generate the expected run_name matching train_hsi_hashing.py
    run_name = f"{model}_{dataset}_{loss}_pca{pca}_patch{patch}_bits{bit}"
    report_path = os.path.join(output_dir, f"{run_name}_retrieval_report.txt")
    npz_path = os.path.join(output_dir, f"{run_name}_pr.npz")
    
    # Parse the report
    result = {
        "Dataset": dataset,
        "Loss": loss,
        "Bits": bit,
        "Model": model,
        "mAP": None,
        "Train_Time": None,
        "Eval_Time": None,
        "NPZ_Path": npz_path if os.path.exists(npz_path) else None
    }
    
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            for line in f:
                if "mAP (%)" in line:
                    result["mAP"] = float(line.split(":")[1].strip())
                elif "Training time" in line:
                    result["Train_Time"] = line.split(":")[1].strip().replace(" s", "")
                elif "Eval time" in line:
                    result["Eval_Time"] = line.split(":")[1].strip().replace(" s", "")
    else:
        print(f"Warning: Report not found at {report_path}")
        
    return result

def plot_pr_grids(results_df, output_dir):
    # Professional Publication-Ready Aesthetics
    plt.rcParams.update({
        'font.size': 22,
        'axes.labelsize': 24,
        'axes.titlesize': 26,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.figsize': (10, 8),
        'axes.linewidth': 2.0,
        'lines.linewidth': 3.5,
        'lines.markersize': 12
    })
    
    # Distinct, attractive color palette
    model_colors = {
        "ssftt": "#1f77b4",     # Deep blue
        "mamba": "#ff7f0e",     # Vibrant orange
        "moe_mamba": "#2ca02c", # Rich green
        "ssrn": "#9467bd",      # Purple
        "a2s2kresnet": "#8c564b", # Brown
        "contextualnet": "#e377c2", # Pink
        "cnn": "#d62728",       # Strong red (fallback)
        "cnn2d": "#8c564b",
        "cnn3d": "#17becf",
        "hybridsn": "#7f7f7f",
        "morphformer": "#bcbd22",
        "spectralformer": "#17becf"
    }
    
    # Professional display names for the legend
    model_display_names = {
        "ssftt": "SSFTT",
        "mamba": "Mamba",
        "moe_mamba": "MoE-Mamba",
        "ssrn": "SSRN",
        "a2s2kresnet": "A2S2K-ResNet",
        "contextualnet": "ContextualNet",
        "cnn2d": "CNN-2D",
        "cnn3d": "CNN-3D",
        "hybridsn": "HybridSN",
        "morphformer": "MorphFormer",
        "spectralformer": "SpectralFormer"
    }
    
    groups = results_df.groupby(["Dataset", "Loss", "Bits"])
    
    for (dataset, loss, bits), group in groups:
        plt.figure()
        
        valid_plots = 0
        markers = ['D', 'o', 's', '^', 'v']
        
        for i, (_, row) in enumerate(group.iterrows()):
            model = row["Model"]
            npz = row["NPZ_Path"]
            if npz and os.path.exists(npz):
                data = np.load(npz)
                marker = markers[i % len(markers)]
                color = model_colors.get(model, "#9467bd") # Fallback to purple
                display_name = model_display_names.get(model, model.upper())
                
                plt.plot(data['R'], data['P'], marker=marker, color=color, label=display_name)
                valid_plots += 1
                
        if valid_plots > 0:
            # Subtle dashed grid behind the lines
            plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5, zorder=0)
            
            plt.xlabel('Recall', fontweight='bold')
            plt.ylabel('Precision', fontweight='bold')
            plt.title(f"{dataset.upper()} | {loss.upper()} | {bits} Bits", pad=20)
            
            # Professional legend formatting
            plt.legend(frameon=True, edgecolor='black', fancybox=False, shadow=False)
            
            # Tight layout removes excess white space
            plt.tight_layout()
            
            out_path_pdf = os.path.join(output_dir, f"grid_{dataset}_{loss}_bits{bits}_pr.pdf")
            out_path_png = os.path.join(output_dir, f"grid_{dataset}_{loss}_bits{bits}_pr.png")
            plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
            plt.savefig(out_path_png, format='png', bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved publication-ready PR grid -> {out_path_pdf} and {out_path_png}")

def generate_markdown_table(df, output_path):
    with open(output_path, 'w') as f:
        f.write("# Comprehensive Hashing Comparison\n\n")
        
        for dataset in df["Dataset"].unique():
            f.write(f"## Dataset: {dataset}\n\n")
            
            ds_df = df[df["Dataset"] == dataset]
            
            losses = ds_df["Loss"].unique()
            bits = sorted(ds_df["Bits"].unique())
            
            # Header
            f.write("| Model | " + " | ".join([f"{loss} ({b}b)" for loss in losses for b in bits]) + " | Avg Eval/Retrieval Time (s) |\n")
            f.write("|---|" + "|".join(["---" for _ in range(len(losses) * len(bits))]) + "|---|\n")
            
            model_display_names = {
                "ssftt": "SSFTT",
                "mamba": "Mamba",
                "moe_mamba": "MoE-Mamba",
                "ssrn": "SSRN",
                "a2s2kresnet": "A2S2K-ResNet",
                "contextualnet": "ContextualNet",
                "cnn2d": "CNN-2D",
                "cnn3d": "CNN-3D",
                "hybridsn": "HybridSN",
                "morphformer": "MorphFormer",
                "spectralformer": "SpectralFormer"
            }
            
            models = ds_df["Model"].unique()
            for model in models:
                display_name = model_display_names.get(model, model.upper())
                row_str = f"| {display_name} | "
                eval_times = []
                for loss in losses:
                    for b in bits:
                        val = ds_df[(ds_df["Model"] == model) & (ds_df["Loss"] == loss) & (ds_df["Bits"] == b)]
                        if not val.empty and val['mAP'].values[0] is not None:
                            row_str += f"{val['mAP'].values[0]:.2f} | "
                            if val['Eval_Time'].values[0] is not None:
                                eval_times.append(float(val['Eval_Time'].values[0]))
                        else:
                            row_str += "- | "
                            
                avg_eval = np.mean(eval_times) if eval_times else 0.0
                row_str += f"{avg_eval:.2f} |\n"
                f.write(row_str)
                
            f.write("\n")
    print(f"Generated Markdown table -> {output_path}")

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    # Generate all combinations
    combinations = list(itertools.product(args.datasets, args.losses, args.models, args.bits))
    
    print(f"Total experiments to run: {len(combinations)}")
    
    for (ds, loss, mod, bit) in combinations:
        res = run_experiment(ds, mod, loss, bit, args.epochs, args.patch, args.pca, args.output_dir)
        all_results.append(res)
        
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, "comprehensive_results.csv"), index=False)
    print(f"Saved results to CSV -> {os.path.join(args.output_dir, 'comprehensive_results.csv')}")
    
    print("\nGenerating Markdown Table...")
    generate_markdown_table(df, os.path.join(args.output_dir, "comprehensive_results.md"))
    
    print("\nGenerating PR Curves Grids...")
    plot_pr_grids(df, args.output_dir)
    
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
