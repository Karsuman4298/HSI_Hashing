# HSI SSFTT Hashing - Training Commands Guide

This document contains a comprehensive list of commands to run various experiments on the Houston 2013 dataset using the streamlined SSFTT architecture. 

## 1. Core Base Commands

### A. Pre-Patched Dataset (PCA Compressed to 30 Channels)
**Recommended:** Flattens the pre-patched dataset, applies PCA to reduce noise, and skyrockets performance to ~80% mAP.
```bash
PYTHONPATH=cls_SSFTT_IP python3 cls_SSFTT_IP/train_hsi_hashing.py \
  --dataset houston2013 \
  --prepatched_dir cls_SSFTT_IP/houston13_alreadypatched_dataset \
  --hash_bit_length 64 \
  --epochs 100 \
  --pca 30
```

### B. Pre-Patched Dataset (Full 144 Channels)
Uses the pre-extracted 11x11 patches without any PCA compression.
```bash
PYTHONPATH=cls_SSFTT_IP python3 cls_SSFTT_IP/train_hsi_hashing.py \
  --dataset houston2013 \
  --prepatched_dir cls_SSFTT_IP/houston13_alreadypatched_dataset \
  --hash_bit_length 64 \
  --epochs 100 \
  --pca 144
```

### C. Our DBSCAN Disjoint Split (Full 144 Channels)
Uses our custom spatial clustering to completely isolate training/testing pixels geographically to simulate severe real-world domain shifts.
```bash
PYTHONPATH=cls_SSFTT_IP python3 cls_SSFTT_IP/train_hsi_hashing.py \
  --dataset houston2013 \
  --data_dir cls_SSFTT_IP/houston13_our_dataset \
  --patch 11 \
  --split_type disjoint \
  --hash_bit_length 64 \
  --epochs 100 \
  --pca 144
```

---

## Quick Reference Flags
* `--hash_bit_length`: Change the length of the binary hash code (`16`, `32`, `64`, `128`)
* `--pca`: Change the number of channels (`30`, `48`, `144`). Set to `144` or `0` to completely bypass PCA compression.
* `--epochs`: Change the number of training loops (default `100`).
* `--num_tokens`: Change the number of learned tokens for SSFTT compression (default `4`).



"""
# 1. CSQ (Central Similarity Quantization)
PYTHONPATH=cls_SSFTT_IP python3 cls_SSFTT_IP/train_hsi_hashing.py \
  --dataset houston2013 --prepatched_dir cls_SSFTT_IP/houston13_alreadypatched_dataset \
  --hash_bit_length 64 --epochs 100 --pca 30 --loss_type csq

# 2. DPN (Deep Pairwise Hashing)
PYTHONPATH=cls_SSFTT_IP python3 cls_SSFTT_IP/train_hsi_hashing.py \
  --dataset houston2013 --prepatched_dir cls_SSFTT_IP/houston13_alreadypatched_dataset \
  --hash_bit_length 64 --epochs 100 --pca 30 --loss_type dpn

# 3. DSH (Deep Supervised Hashing)
PYTHONPATH=cls_SSFTT_IP python3 cls_SSFTT_IP/train_hsi_hashing.py \
  --dataset houston2013 --prepatched_dir cls_SSFTT_IP/houston13_alreadypatched_dataset \
  --hash_bit_length 64 --epochs 100 --pca 30 --loss_type dsh

# 4. GreedyHash
PYTHONPATH=cls_SSFTT_IP python3 cls_SSFTT_IP/train_hsi_hashing.py \
  --dataset houston2013 --prepatched_dir cls_SSFTT_IP/houston13_alreadypatched_dataset \
  --hash_bit_length 64 --epochs 100 --pca 30 --loss_type greedyhash

# 5. HashNet
PYTHONPATH=cls_SSFTT_IP python3 cls_SSFTT_IP/train_hsi_hashing.py \
  --dataset houston2013 --prepatched_dir cls_SSFTT_IP/houston13_alreadypatched_dataset \
  --hash_bit_length 64 --epochs 100 --pca 30 --loss_type hashnet

# 6. IDHN (Improved Deep Hashing Network)
PYTHONPATH=cls_SSFTT_IP python3 cls_SSFTT_IP/train_hsi_hashing.py \
  --dataset houston2013 --prepatched_dir cls_SSFTT_IP/houston13_alreadypatched_dataset \
  --hash_bit_length 64 --epochs 100 --pca 30 --loss_type idhn

"""