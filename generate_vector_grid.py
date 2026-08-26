import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'lines.linewidth': 4,
    'lines.markersize': 12
})

folder = '/Users/sumankar/Desktop/HSI_SSFTT/cls_SSFTT_IP/all_trento_results'
losses = ['csq', 'dpn', 'dsh', 'greedyhash', 'hashnet', 'idhn', 'orthohash', 'dspch', 'dhnn']
bits = ['16', '32', '64']
models = ['ssftt', 'mamba', 'moe_mamba', 'ssrn', 'a2s2kresnet', 'contextualnet', 'cnn2d', 'cnn3d', 'hybridsn', 'morphformer']

model_colors = {
    "ssftt": "#1f77b4",     
    "mamba": "#ff7f0e",     
    "moe_mamba": "#2ca02c",
    "ssrn": "#9467bd",
    "a2s2kresnet": "#8c564b",
    "contextualnet": "#e377c2",
    "cnn2d": "#8c564b",
    "cnn3d": "#17becf",
    "hybridsn": "#7f7f7f",
    "morphformer": "#bcbd22"
}

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
    "morphformer": "MorphFormer"
}
markers = ['D', 'o', 's', '^', 'v', 'p', '*', 'h', 'x', '+']

# Change figsize to accurately reflect the 10x8 aspect ratio of the original individual plots!
fig, axes = plt.subplots(3, 9, figsize=(90, 24))

for r, bit in enumerate(bits):
    for c, loss in enumerate(losses):
        ax = axes[r, c]
        for i, model in enumerate(models):
            npz = f"{model}_trento_{loss}_pca30_patch11_bits{bit}_pr.npz"
            path = os.path.join(folder, npz)
            if os.path.exists(path):
                data = np.load(path)
                ax.plot(data['R'], data['P'], marker=markers[i], color=model_colors[model], label=model_display_names[model])
        
        ax.set_title(f"{loss.upper()} ({bit} Bits)", fontweight='bold', pad=15)
        
        if c == 0:
            ax.set_ylabel('Precision', fontweight='bold', labelpad=10)
        if r == 2:
            ax.set_xlabel('Recall', fontweight='bold', labelpad=10)
            
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', frameon=True, edgecolor='black')

plt.tight_layout(pad=3.0)
plt.savefig('/Users/sumankar/Desktop/HSI_SSFTT/Massive_PR_Comparison_Grid_Vector_Fixed.pdf', format='pdf', bbox_inches='tight')
print("Successfully generated Fixed Vector PDF!")
