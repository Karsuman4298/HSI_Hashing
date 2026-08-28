# HSI Hashing - Comprehensive Command Guide

This guide covers everything from training individual models, running massive grid searches across all architectures and loss functions, to generating publication-ready visualizations.

---

## 1. Individual Model Training

Use `train_hsi_hashing.py` to train a single configuration. This script supports 9 hashing loss functions and 10 architectures.

### Basic Training Command (Example: Houston 2013, SSFTT, CSQ, 64-bit)
```bash
python3 cls_SSFTT_IP/train_hsi_hashing.py \
  --dataset houston2013 \
  --model ssftt \
  --loss_type csq \
  --hash_bit_length 64 \
  --epochs 100 \
  --pca 30 \
  --output_dir output
```

### Dataset Options
- `houston2013`
- `houston2018`
- `trento`
- `nilifossae`
- `indian_pines`

### Supported Models
`ssftt`, `mamba`, `moe_mamba`, `ssrn`, `a2s2kresnet`, `contextualnet`, `cnn2d`, `cnn3d`, `hybridsn`, `morphformer`

### Supported Loss Functions
`csq`, `dpn`, `dsh`, `greedyhash`, `hashnet`, `idhn`, `orthohash`, `dspch`, `dhnn`

---

## 2. Comprehensive Study Orchestration

To run a massive batch of models across multiple datasets, bits, and loss functions automatically, use `run_comprehensive_study.py`. This script coordinates `train_hsi_hashing.py` calls and saves everything cleanly into specific folders.

### Example: Train 6 Models on Trento across All Bits and Losses
```bash
python3 run_comprehensive_study.py \
  --datasets trento \
  --models ssftt mamba moe_mamba ssrn a2s2kresnet contextualnet \
  --losses csq dhnn dpn dsh dspch greedyhash hashnet idhn orthohash \
  --bits 16 32 64 \
  --epochs 100 \
  --output_dir trained_results_hasing/output_Trento_Comprehensive
```

*Note: Results (params, PR curves, logs) will be saved in the specified `--output_dir`.*

---

## 3. Master Visualization (Precision-Recall Curves)

Use `master_visualization.py` with `--mode pr_curve` to combine multiple models into single publication-ready PR plots. 

If your results are split across multiple directories (e.g., `output_Trento_ssft_mamba_moemamba` and `output_Trento_A2S2Kresnet_ssrn_ContextualNet`), you can pass **multiple folders** to `--result_dir`.

### Example: Generate Combined 6-Model PR Curves for Houston 2013
This command instantly generates PR curves for all 9 loss functions and all 3 bit lengths by automatically scanning the provided result folders:

```bash
BASE_DIR="trained_results_hasing"
OUT_DIR="master_visualizations"

python3 master_visualization.py \
  --mode pr_curve \
  --models ssftt mamba moe_mamba ssrn a2s2kresnet contextualnet \
  --datasets houston2013 \
  --losses csq dhnn dpn dsh dspch greedyhash hashnet idhn orthohash \
  --bits 16 32 64 \
  --result_dir "$BASE_DIR/output_Houston13_ssft_mamba_moemamba" "$BASE_DIR/Houston13_SSRN_A2S2KResNet_Contextualnet" \
  --output_dir "$OUT_DIR"
```

---

## 4. Master Visualization (Classification Maps)

Use `master_visualization.py` with `--mode cls_map` to generate false-color classification maps and confusion matrices from saved model weights (`.pth` files).

### Example: Generate a Classification Map
```bash
python3 master_visualization.py \
  --mode cls_map \
  --models ssftt \
  --datasets houston2013 \
  --losses csq \
  --bits 64 \
  --result_dir trained_results_hasing/output_Houston13_ssft_mamba_moemamba \
  --output_dir master_visualizations
```

### Advanced Flags for Classification Maps
- `--block_background`: Sets the unlabeled background pixels to completely black.
- `--mask_background`: Forces the Ground Truth background pixels to 0, completely ignoring them in visualization.

Example with background blocking:
```bash
python3 master_visualization.py \
  --mode cls_map \
  --models mamba \
  --datasets trento \
  --losses dhnn \
  --bits 32 \
  --result_dir trained_results_hasing/output_Trento_ssft_mamba_moemamba \
  --output_dir master_visualizations \
  --block_background
```