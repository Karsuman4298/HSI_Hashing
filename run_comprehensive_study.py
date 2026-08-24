import os
import subprocess
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive Ablation Study Orchestrator")
    parser.add_argument("--datasets", nargs='+', default=["houston2013"])
    parser.add_argument("--models", nargs='+', default=["ssftt", "mamba", "moe_mamba"])
    parser.add_argument("--losses", nargs='+', default=["csq", "dpn", "dsh", "greedyhash", "hashnet", "idhn", "orthohash", "dspch", "dhnn"])
    parser.add_argument("--bits", nargs='+', type=int, default=[16, 32, 64])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patch", type=int, default=11)
    parser.add_argument("--pca", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="output")
    return parser.parse_args()

def run_experiment(dataset, model, loss, bit, epochs, patch, pca, output_dir):
    # Known prepatched directories. Add more if your structure changes.
    prepatched_map = {
        "houston2013": "cls_SSFTT_IP/houston13_alreadypatched_dataset",
        "trento": "cls_SSFTT_IP/trento_alreadypatched_dataset",
        "indian_pines": "cls_SSFTT_IP/ip_alreadypatched_dataset"
    }
    
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
    
    if dataset in prepatched_map and os.path.exists(prepatched_map[dataset]):
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
    # Group by dataset, loss, bits
    # Plot SSFTT, Mamba, MoE-Mamba on same graph
    groups = results_df.groupby(["Dataset", "Loss", "Bits"])
    
    for (dataset, loss, bits), group in groups:
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(8, 6))
        
        valid_plots = 0
        markers = ['D', 'o', 's', '^', 'v']
        
        for i, (_, row) in enumerate(group.iterrows()):
            model = row["Model"]
            npz = row["NPZ_Path"]
            if npz and os.path.exists(npz):
                data = np.load(npz)
                marker = markers[i % len(markers)]
                plt.plot(data['R'], data['P'], marker=marker, label=model, linewidth=2, markersize=8)
                valid_plots += 1
                
        if valid_plots > 0:
            plt.grid(True)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f"{dataset} | {loss} | {bits} bits")
            plt.legend()
            
            out_path = os.path.join(output_dir, f"grid_{dataset}_{loss}_bits{bits}_pr.pdf")
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            print(f"Saved combined PR grid -> {out_path}")

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
            
            models = ds_df["Model"].unique()
            for model in models:
                row_str = f"| {model} | "
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
