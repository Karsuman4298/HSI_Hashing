# Hyperspectral Image (HSI) Hashing Framework

This repository is a comprehensive benchmarking framework for **Hyperspectral Image (HSI) Hashing**. 
It builds upon the original SSFTT classification framework and adapts various state-of-the-art neural network architectures to perform HSI retrieval and hashing, evaluating them across multiple HSI datasets using diverse hashing loss functions.

---

## 1. Project Overview and Motivation

While classification assigns each pixel to a specific land-cover category, **Hashing** generates compact, binary representations (hash codes) for each pixel. These hash codes allow for extremely fast and storage-efficient retrieval of similar hyperspectral signatures from massive databases.

This framework transforms traditional HSI classification models into HSI hashing networks by standardizing their heads. Every backbone in this repository has been modified to output a 1024-dimensional continuous vector, which is then projected into the final `K`-bit hash code (e.g., 16, 32, or 64 bits).

### Original Reference
This repository originally started as the PyTorch implementation of the **Spectral–Spatial Feature Tokenization Transformer (SSFTT)**:
> L. Sun, G. Zhao, Y. Zheng and Z. Wu, "Spectral–Spatial Feature Tokenization Transformer for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, 2022.

## 2. Core Components

### 2.1 Supported Datasets
The framework is designed to work with disjoint pre-patched subsets of the following popular datasets:
- **Houston 2013** (`houston2013`)
- **Houston 2018** (`houston2018`)
- **Trento** (`trento`)
- **Indian Pines** (`indian_pines`)
- **Nili Fossae (Mars)** (`nilifossae`)

*Data loading is streamlined to expect a pre-patched `.mat` directory structure containing `HSI_Tr.mat`, `HSI_Te.mat`, `TrLabel.mat`, and `TeLabel.mat`.*

### 2.2 Model Backbones (`--model`)
We have integrated a diverse array of models. Each model strips away its original classification layer and employs a standardized Hashing Head `[Dropout(0.5) -> Linear(1024) -> ReLU -> Linear(bits)]` to ensure fair comparisons.

1. **`ssftt` (Spectral-Spatial Feature Tokenization Transformer)**: Uses 3D/2D CNNs for feature extraction followed by a Gaussian-weighted tokenizer and a Transformer encoder.
2. **`cnn` (CNN Baseline)**: A standard CNN network used as a performance baseline.
3. **`mamba`**: Leverages State Space Models (SSMs) to capture long-range spectral-spatial dependencies natively.
4. **`moe_mamba`**: A Mixture-of-Experts (MoE) variant of Mamba, employing a router to combine different expert sub-networks for richer representations.
5. **`ssrn` (Spectral-Spatial Residual Network)**: Extracts deep spatial and spectral features using continuous 3D residual blocks.
6. **`a2s2kresnet` (Attention-Based Adaptive Spectral-Spatial Kernel ResNet)**: Utilizes dynamic attention kernels to capture complex HSI characteristics before passing them into a 3D residual backbone.
7. **`contextualnet` (Contextual Deep CNN)**: Based on Lee et al.'s IGARSS 2016 model, using inception modules and residual learning. We adapted it with an `AdaptiveAvgPool2d(1)` layer to support arbitrary HSI patch sizes (like 11x11).

### 2.3 Hashing Loss Functions (`--losses`)
To train the models to output distinct, separable binary codes, the framework supports a wide variety of retrieval loss functions:
- **CSQ** (Center Similarity Quantization)
- **DPN** (Deep Polarized Network)
- **DSH** (Deep Supervised Hashing)
- **GreedyHash**
- **HashNet**
- **IDHN** (Improved Deep Hashing Network)
- **OrthoHash**
- **DSPCH**
- **DHNN** (Deep Hashing Neural Network)

## 3. How It Works

### The Hashing Pipeline
1. **Data Loading**: Pre-patched datasets are loaded. PCA is applied dynamically (if requested) to reduce the spectral band dimension (default is 30 PCA channels, though models like SSRN/ContextualNet often work directly on the raw bands).
2. **Forward Pass**: The selected backbone extracts spectral-spatial features and outputs raw continuous logits (size `bits`).
3. **Loss Calculation**: The chosen hashing loss function evaluates how well these logits separate different classes and forces them towards `-1` or `+1`.
4. **Evaluation**: Once trained, the network generates binary codes for a query set and a database set. We calculate the **mean Average Precision (mAP)** by performing Hamming distance-based retrieval.

## 4. Running the Code

### 4.1 Single Experiment
You can run a single model configuration using the main training script:
```bash
python3 cls_SSFTT_IP/train_hsi_hashing.py \
  --dataset houston2013 \
  --model ssrn \
  --loss_type csq \
  --hash_bit_length 16 \
  --epochs 100 \
  --patch 11 \
  --pca 30 \
  --output_dir output \
  --prepatched_dir ../houston13
```

### 4.2 The Comprehensive Orchestrator
To run mass ablation studies across multiple models, datasets, losses, and bits, use the `run_comprehensive_study.py` orchestrator:

```bash
python3 run_comprehensive_study.py \
  --datasets houston2013 nilifossae \
  --models ssftt mamba moe_mamba ssrn a2s2kresnet contextualnet \
  --losses csq dpn \
  --bits 16 32 64 \
  --epochs 100 \
  --patch 11 \
  --pca 30 \
  --output_dir comprehensive_results
```
This orchestrator will sequentially spin up subprocesses for every combination of the parameters you provide.

## 5. Artifacts and Outputs
When running experiments, the framework will generate the following in your `--output_dir`:
- **Model Checkpoints (`.pth`)**: The best-performing model weights.
- **Retrieval Reports (`.txt`)**: Text files containing mAP metrics and exact precision-recall data.
- **Precision-Recall Data (`.npz`)**: Raw numpy arrays of the PR curves.
- **Precision-Recall Plots (`.pdf` / `.png`)**: Visual charts plotting Recall (x-axis) vs Precision (y-axis).
- **Aggregated CSV (`comprehensive_results.csv`)**: A summary file containing all mAP and training times.
- **Markdown Table (`comprehensive_results.md`)**: A nicely formatted table for easy viewing of the final results.

*(Note: Apple Silicon users running locally might need to prepend `PYTORCH_ENABLE_MPS_FALLBACK=1` when running models like ContextualNet that rely on LocalResponseNorm, which is not yet fully supported on MPS. Linux servers with CUDA do not require this.)*
