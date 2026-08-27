"""
Generalized HSI Classification Training Script (SSFTT)
=======================================================
Works with: Indian Pines, Pavia University, Pavia Center, Salinas,
            Houston 2013, Houston 2018, Kennedy Space Center, Botswana,
            WHU-Hi LongKou, Hanchuan, Honghu, and any custom HSI .mat pair.

Usage:
    python train_hsi.py --dataset pavia_university
    python train_hsi.py --dataset indian_pines --pca 30 --patch 13 --test_ratio 0.9
    python train_hsi.py --dataset custom \
        --data_file MyData.mat --data_key my_data \
        --label_file MyGT.mat  --label_key my_gt \
        --class_names "Tree,Water,Road"
"""

import os
import time
import argparse
import random
import h5py
import numpy as np
import p2lc
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
)
from operator import truediv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import seaborn as sns
import matplotlib.pyplot as plt

from SSFTTHashNet import SSFTTHashNet, CNNBaselineHashNet
from hash_losses import CSQLoss, DPNLoss, SupConLoss, DSHLoss, GreedyHashLoss, HashNetLoss, IDHNLoss, OrthoHashLoss, DSPCHLoss, BatchDHNNLoss
import get_cls_map  # existing visualization module
from get_cls_map import get_classification_map, list_to_colormap, classification_map


# ──────────────────────────────────────────────────────────────
# 1.  DATASET REGISTRY
# ──────────────────────────────────────────────────────────────
# Add your own datasets here.  Only `data_file`, `data_key`,
# `label_file`, `label_key`, and `class_names` are required.

DATASETS = {

    # ── Indian Pines ──────────────────────────────────────────
    "indian_pines": {
        "data_file":  "Indian_pines_corrected.mat",
        "data_key":   "indian_pines_corrected",
        "label_file": "Indian_pines_gt.mat",
        "label_key":  "indian_pines_gt",
        "class_names": [
            "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
            "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
            "Hay-windrowed", "Oats", "Soybean-notill",
            "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
            "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
        ],
    },

    # ── Pavia University ──────────────────────────────────────
    "pavia_university": {
        "data_file":  "PaviaU.mat",
        "data_key":   "paviaU",
        "label_file": "PaviaU_gt.mat",
        "label_key":  "PaviaU_gt",
        "class_names": [
            "Asphalt", "Meadows", "Gravel", "Trees",
            "Painted-metal-sheets", "Bare-Soil", "Bitumen",
            "Self-Blocking-Bricks", "Shadows",
        ],
    },

    # ── Pavia Center ──────────────────────────────────────────
    "pavia_center": {
        "data_file":  "Pavia.mat",
        "data_key":   "pavia",
        "label_file": "Pavia_gt.mat",
        "label_key":  "pavia_gt",
        "class_names": [
            "Water", "Trees", "Asphalt", "Self-Blocking-Bricks",
            "Bitumen", "Tiles", "Shadows", "Meadows", "Bare-Soil",
        ],
    },

    # ── Salinas ───────────────────────────────────────────────
    "salinas": {
        "data_file":  "Salinas_corrected.mat",
        "data_key":   "salinas_corrected",
        "label_file": "Salinas_gt.mat",
        "label_key":  "Salinas_gt",
        "class_names": [
            "Brocoli_green_weeds_1", "Brocoli_green_weeds_2",
            "Fallow", "Fallow_rough_plow", "Fallow_smooth",
            "Stubble", "Celery", "Grapes_untrained",
            "Soil_vinyard_develop", "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk", "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk", "Lettuce_romaine_7wk",
            "Vinyard_untrained", "Vinyard_vertical_trellis",
        ],
    },

    # ── Houston 2013 ─────────────────────────────────────────
    "houston2013": {
        "data_file":  "houston13_data.mat",
        "data_key":   "data",
        "label_file": "houston13_gt.mat",
        "label_key":  "gt",
        "class_names": [
            "Healthy-grass", "Stressed-grass", "Synthetic-grass",
            "Trees", "Soil", "Water", "Residential", "Commercial",
            "Road", "Highway", "Railway", "Parking-lot-1",
            "Parking-lot-2", "Tennis-court", "Running-track",
        ],
    },

    # ── Trento ───────────────────────────────────────────────
    "trento": {
        "data_file":  "trento_data.mat",
        "data_key":   "data",
        "label_file": "trento_gt.mat",
        "label_key":  "gt",
        "class_names": [
            "Apples", "Buildings", "Ground", "Woods", "Vineyard", "Roads"
        ],
    },

    # ── Houston 2018 (GRSS DFC) ──────────────────────────────
    "houston2018": {
        "data_file":  "Houston2018.mat",
        "data_key":   "Houston2018",
        "label_file": "Houston2018_gt.mat",
        "label_key":  "Houston2018_gt",
        "class_names": [
            "Healthy-grass", "Stressed-grass", "Artificial-turf",
            "Evergreen-trees", "Deciduous-trees", "Bare-earth",
            "Water", "Residential-buildings",
            "Non-residential-buildings", "Roads", "Sidewalks",
            "Crosswalks", "Major-thoroughfares", "Highways",
            "Railways", "Paved-parking", "Unpaved-parking",
            "Cars", "Trains", "Stadium-seats",
        ],
    },

    # ── Nili Fossae ──────────────────────────────────────────
    "nilifossae": {
        "data_file":  "NiliFossae.mat",
        "data_key":   "NiliFossae",
        "label_file": "NiliFossae_gt.mat",
        "label_key":  "NiliFossae_gt",
        "class_names": [f"Class {i+1}" for i in range(9)]
    },

    # ── Kennedy Space Center ──────────────────────────────────
    "ksc": {
        "data_file":  "KSC.mat",
        "data_key":   "KSC",
        "label_file": "KSC_gt.mat",
        "label_key":  "KSC_gt",
        "class_names": [
            "Scrub", "Willow-swamp", "CP-hammock", "Slash-pine",
            "Oak/Broadleaf", "Hardwood", "Swamp", "Graminoid-marsh",
            "Spartina-marsh", "Cattail-marsh", "Salt-marsh",
            "Mud-flats", "Water",
        ],
    },

    # ── Botswana ──────────────────────────────────────────────
    "botswana": {
        "data_file":  "Botswana.mat",
        "data_key":   "Botswana",
        "label_file": "Botswana_gt.mat",
        "label_key":  "Botswana_gt",
        "class_names": [
            "Water", "Hippo-grass", "Floodplain-grass-1",
            "Floodplain-grass-2", "Reeds", "Riparian",
            "Firescar", "Island-interior", "Acacia-woodlands",
            "Acacia-shrublands", "Acacia-grasslands",
            "Short-mopane", "Mixed-mopane", "Exposed-soils",
        ],
    },

    # ── WHU-Hi LongKou ────────────────────────────────────────
    "longkou": {
        "data_file":  "WHU_Hi_LongKou.mat",
        "data_key":   "WHU_Hi_LongKou",
        "label_file": "WHU_Hi_LongKou_gt.mat",
        "label_key":  "WHU_Hi_LongKou_gt",
        "class_names": [
            "Corn", "Cotton", "Sesame", "Broad-leaf soybean",
            "Narrow-leaf soybean", "Rice", "Water",
            "Roads and houses", "Mixed weed",
        ],
    },

    # ── WHU-Hi HanChuan ───────────────────────────────────────
    "hanchuan": {
        "data_file":  "WHU_Hi_HanChuan.mat",
        "data_key":   "WHU_Hi_HanChuan",
        "label_file": "WHU_Hi_HanChuan_gt.mat",
        "label_key":  "WHU_Hi_HanChuan_gt",
        "class_names": [
            "Strawberry", "Cowpea", "Soybean", "Sorghum", 
            "Water spinach", "Watermelon", "Greens", "Trees", 
            "Grass", "Red roof", "Gray roof", "Plastic", 
            "Bare soil", "Road", "Bright object", "Water"
        ],
    },

    # ── WHU-Hi HongHu ─────────────────────────────────────────
    "honghu": {
        "data_file":  "WHU_Hi_HongHu.mat",
        "data_key":   "WHU_Hi_HongHu",
        "label_file": "WHU_Hi_HongHu_gt.mat",
        "label_key":  "WHU_Hi_HongHu_gt",
        "class_names": [
            "Red roof", "Road", "Bare soil", "Cotton", 
            "Cotton firewood", "Rape", "Chinese cabbage", "Pakchoi", 
            "Cabbage", "Tuber mustard", "Brassica parachinensis", "Brassica chinensis", 
            "Small Brassica chinensis", "Lactuca sativa", "Celtuce", "Film covered lettuce", 
            "Romaine lettuce", "Carrot", "White radish", "Garlic sprout", 
            "Broad bean", "Tree"
        ],
    },
}


# ──────────────────────────────────────────────────────────────
# 2.  DATA LOADING (Supports both standard .mat and MATLAB v7.3)
# ──────────────────────────────────────────────────────────────

def _find_mat_key(mat_dict, preferred_key):
    """Return preferred key or find the first array key."""
    if preferred_key and preferred_key in mat_dict:
        return preferred_key
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and not k.startswith("__"):
            return k
    raise KeyError("No usable array found in .mat file.")


def load_single_mat(file_path, preferred_key=None, is_label=False):
    """
    Robust reader: Automatically detects whether the file is standard 
    MATLAB format or MATLAB v7.3 (HDF5).
    """
    try:
        # 1. Try standard SciPy loadmat (MATLAB <= v7)
        mat_dict = sio.loadmat(file_path)
        if is_label:
            valid_keys = [k for k in mat_dict.keys() if not k.startswith('__') and ('gt' in k.lower() or 'label' in k.lower())]
            key = valid_keys[0] if valid_keys else _find_mat_key(mat_dict, preferred_key)
        else:
            key = _find_mat_key(mat_dict, preferred_key)
        return np.array(mat_dict[key])
    except (NotImplementedError, Exception):
        # 2. Fallback to h5py for MATLAB v7.3 files
        try:
            with h5py.File(file_path, 'r') as f:
                if preferred_key and preferred_key in f:
                    data = np.array(f[preferred_key])
                else:
                    valid_keys = [k for k in f.keys() if not k.startswith('#')]
                    if not valid_keys:
                        raise KeyError(f"No valid dataset found in {file_path}")
                        
                    if is_label:
                        label_keys = [k for k in valid_keys if 'gt' in k.lower() or 'label' in k.lower()]
                        if label_keys:
                            data = np.array(f[label_keys[0]])
                        else:
                            # Usually the smallest array is the label
                            data = np.array(f[valid_keys[1]] if len(valid_keys) > 1 else f[valid_keys[0]])
                    else:
                        # Find the first key that isn't metadata (usually the large data array)
                        data_keys = [k for k in valid_keys if 'gt' not in k.lower() and 'label' not in k.lower()]
                        data = np.array(f[data_keys[0]] if data_keys else f[valid_keys[0]])
                
                # MATLAB v7.3 saves arrays in reverse dimension order.
                # However, if the file was generated natively in Python (h5py), it might already be correct.
                # We check the shape: if the first dimension is small (e.g. C) and the last is large (e.g. N), we transpose.
                if len(data.shape) == 4 and data.shape[0] < data.shape[-1]:
                    data = data.T
                elif len(data.shape) == 3 and data.shape[0] < data.shape[-1]:
                    data = data.T
                return data
        except Exception:
            # 3. Fallback to numpy load (in case it's actually an .npy file renamed to .mat)
            data = np.load(file_path, allow_pickle=True)
            # If the numpy array is just a wrapper around a dictionary, extract the array
            if data.shape == () and isinstance(data.item(), dict):
                mat_dict = data.item()
                key = _find_mat_key(mat_dict, preferred_key)
                return np.array(mat_dict[key])
            return data


def loadData(data_dir, ds_cfg):
    """Load an HSI data cube and its ground-truth label map."""
    data_path  = os.path.join(data_dir, ds_cfg["data_file"])
    label_path = os.path.join(data_dir, ds_cfg["label_file"])

    print(f"Loading data from : {data_path}")
    print(f"Loading label from: {label_path}")

    data   = load_single_mat(data_path,  ds_cfg.get("data_key"))
    labels = load_single_mat(label_path, ds_cfg.get("label_key"))

    # Ensure labels are 2D: (H, W)
    if labels.ndim == 3 and labels.shape[-1] == 1:
        labels = labels.squeeze(-1)

    return data, labels
# ──────────────────────────────────────────────────────────────
# 3.  PREPROCESSING  (unchanged logic, just cleaner)
# ──────────────────────────────────────────────────────────────

def applyPCA(X, numComponents):
    """Reduce the spectral dimension with PCA (whitened)."""
    h, w, bands = X.shape
    X_flat = X.reshape(-1, bands)
    pca = PCA(n_components=numComponents, whiten=True)
    X_pca = pca.fit_transform(X_flat)
    return X_pca.reshape(h, w, numComponents)


def padWithZeros(X, margin=2):
    """Zero-pad the spatial borders so every pixel gets a full patch."""
    h, w, c = X.shape
    padded = np.zeros((h + 2 * margin, w + 2 * margin, c), dtype=X.dtype)
    padded[margin:margin + h, margin:margin + w, :] = X
    return padded


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    """Extract a (windowSize × windowSize) patch around every pixel."""
    margin = (windowSize - 1) // 2
    padded = padWithZeros(X, margin)

    h, w = X.shape[:2]
    n_pixels = h * w
    patches = np.zeros((n_pixels, windowSize, windowSize, X.shape[2]),
                       dtype=X.dtype)
    labels  = np.zeros(n_pixels, dtype=y.dtype)

    idx = 0
    for r in range(margin, padded.shape[0] - margin):
        for c in range(margin, padded.shape[1] - margin):
            patches[idx] = padded[r - margin:r + margin + 1,
                                  c - margin:c + margin + 1]
            labels[idx]  = y[r - margin, c - margin]
            idx += 1

    if removeZeroLabels:
        mask = labels > 0
        patches = patches[mask]
        labels  = labels[mask] - 1          # 0-indexed for CrossEntropyLoss

    return patches, labels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    return train_test_split(
        X, y, test_size=testRatio,
        random_state=randomState, stratify=y,
    )


# ──────────────────────────────────────────────────────────────
# 4.  PYTORCH DATASETS
# ──────────────────────────────────────────────────────────────

class HSIDataset(torch.utils.data.Dataset):
    """Single Dataset class for both train and test splits."""

    def __init__(self, X, y, augment=False, weights=None):
        self.x = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.weights = torch.FloatTensor(weights) if weights is not None else torch.ones(len(y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img = self.x[index]
        label = self.y[index]
        weight = self.weights[index]

        if self.augment:
            # Spatial augmentations (flips and rotations)
            # img shape: (1, bands, H, W)
            if random.random() > 0.5:
                img = torch.flip(img, dims=[-1]) # Horizontal flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[-2]) # Vertical flip
                
            k = random.randint(0, 3)
            if k > 0:
                img = torch.rot90(img, k, dims=[-2, -1])
                
            # Spectral augmentation (Gaussian noise & Band masking)
            if random.random() > 0.5:
                noise = torch.randn_like(img) * 0.01
                img = img + noise
                
            if random.random() > 0.5:
                # Random band masking (zero out ~10% of spectral bands)
                mask = (torch.rand(img.shape[1]) > 0.1).float().view(1, -1, 1, 1)
                img = img * mask

        return img, label, weight, index


# ──────────────────────────────────────────────────────────────
# 5.  DATA-LOADER FACTORY
# ──────────────────────────────────────────────────────────────

def create_data_loader(args):
    """End-to-end: raw .mat → train / test / all DataLoaders."""

    ds_cfg = DATASETS[args.dataset]
    num_classes = len(ds_cfg["class_names"])

    if hasattr(args, 'prepatched_dir') and args.prepatched_dir is not None:
        import scipy.io as sio
        import glob
        
        # Try to dynamically find the tr and te files (to support custom names like WHU_Hi_HanChuan_Tr.mat)
        if args.pca == 48:
            tr_file = "HSI_Tr_48.mat"
            te_file = "HSI_Te_48.mat"
        else:
            tr_file = "HSI_Tr.mat"
            te_file = "HSI_Te.mat"
            
        # Check if default files exist, if not search for *_Tr.mat and *_Te.mat
        if not os.path.exists(os.path.join(args.prepatched_dir, tr_file)):
            tr_candidates = glob.glob(os.path.join(args.prepatched_dir, "*_Tr.mat"))
            te_candidates = glob.glob(os.path.join(args.prepatched_dir, "*_Te.mat"))
            if tr_candidates and te_candidates:
                tr_file = os.path.basename(tr_candidates[0])
                te_file = os.path.basename(te_candidates[0])

        print(f"\n... Loading Pre-Patched Disjoint Split from {args.prepatched_dir} ({tr_file}) ...")
        Xtrain = load_single_mat(os.path.join(args.prepatched_dir, tr_file))
        Xtest  = load_single_mat(os.path.join(args.prepatched_dir, te_file))
        
        tr_label_path = os.path.join(args.prepatched_dir, "TrLabel.mat")
        te_label_path = os.path.join(args.prepatched_dir, "TeLabel.mat")
        
        # Look for WHU-Hi style custom names if TrLabel.mat doesn't exist
        if not os.path.exists(tr_label_path):
            tr_gt_candidates = glob.glob(os.path.join(args.prepatched_dir, "*_Tr_gt.mat"))
            te_gt_candidates = glob.glob(os.path.join(args.prepatched_dir, "*_Te_gt.mat"))
            if tr_gt_candidates and te_gt_candidates:
                tr_label_path = tr_gt_candidates[0]
                te_label_path = te_gt_candidates[0]
        
        # Fallback logic: If separate label files don't exist, try to extract them from the main data file 
        # (e.g. if 'Houston18_Tr_gt' is bundled inside 'Houston18_Tr.mat')
        if os.path.exists(tr_label_path):
            ytrain = load_single_mat(tr_label_path, preferred_key=None, is_label=True).squeeze()
        else:
            print(f"Warning: {tr_label_path} not found. Attempting to extract bundled labels from {tr_file}...")
            ytrain = load_single_mat(os.path.join(args.prepatched_dir, tr_file), preferred_key=None, is_label=True).squeeze()
            
        if os.path.exists(te_label_path):
            ytest = load_single_mat(te_label_path, preferred_key=None, is_label=True).squeeze()
        else:
            print(f"Warning: {te_label_path} not found. Attempting to extract bundled labels from {te_file}...")
            ytest = load_single_mat(os.path.join(args.prepatched_dir, te_file), preferred_key=None, is_label=True).squeeze()
        
        # ── Dynamic PCA for Pre-Patched Data ─────────────────────
        if args.pca > 0 and args.pca < Xtrain.shape[3]:
            print(f"\n... Dynamically applying PCA to patches ({Xtrain.shape[3]} -> {args.pca} channels) ...")
            from sklearn.decomposition import PCA
            
            N_tr, H, W, C = Xtrain.shape
            N_te = Xtest.shape[0]
            
            # Combine to fit PCA globally (simulating full-image PCA)
            X_all = np.concatenate([Xtrain, Xtest], axis=0)
            X_all_flat = X_all.reshape(-1, C)
            
            pca = PCA(n_components=args.pca, whiten=True)
            X_all_pca_flat = pca.fit_transform(X_all_flat)
            
            X_all_pca = X_all_pca_flat.reshape(N_tr + N_te, H, W, args.pca)
            
            Xtrain = X_all_pca[:N_tr]
            Xtest = X_all_pca[N_tr:]
        elif args.pca == 0 or args.pca == Xtrain.shape[3]:
            print(f"\n... Skipping PCA transformation (using full {Xtrain.shape[3]} channels) ...")
            args.pca = Xtrain.shape[3]
        # ─────────────────────────────────────────────────────────
        
        if np.min(ytrain) == 1:
            ytrain = ytrain - 1
            ytest = ytest - 1
            
        print(f"Train samples: {Xtrain.shape}")
        print(f"Test samples : {Xtest.shape}")
        
        Xtrain = np.transpose(Xtrain, (0, 3, 1, 2))
        Xtest = np.transpose(Xtest, (0, 3, 1, 2))
        Xtrain = np.expand_dims(Xtrain, axis=1)
        Xtest = np.expand_dims(Xtest, axis=1)
        
        # Ensure args.patch is updated to the actual spatial dimension for reporting
        args.patch = Xtrain.shape[-1]
        
        np.random.seed(args.seed)
        indices = np.random.permutation(len(Xtest))
        num_query = int(len(Xtest) * args.query_ratio)
        query_idx = indices[:num_query]
        db_idx = indices[num_query:]
        
        Xquery, yquery = Xtest[query_idx], ytest[query_idx]
        Xdb, ydb = Xtest[db_idx], ytest[db_idx]
        
        print("--- Pre-Patched Data Loader Complete ---")
        train_dataset = HSIDataset(Xtrain, ytrain)
        db_dataset = HSIDataset(Xdb, ydb)
        query_dataset = HSIDataset(Xquery, yquery)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        db_loader = torch.utils.data.DataLoader(db_dataset, batch_size=args.batch_size, shuffle=False)
        query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False)
        return train_loader, db_loader, query_loader, None, num_classes

    # ── Load ──────────────────────────────────────────────────
    X, y = loadData(args.data_dir, ds_cfg)
    print(f"Dataset        : {args.dataset}")
    print(f"HSI shape      : {X.shape}")
    print(f"Label shape    : {y.shape}")
    print(f"Num classes    : {num_classes}")

    # ── PCA ───────────────────────────────────────────────────
    if args.pca == 0 or args.pca == X.shape[2]:
        print("\n... Skipping PCA transformation (using full channels) ...")
        X_pca = X
        args.pca = X.shape[2] # Ensure args.pca has the exact channel count for the network
    else:
        print("\n... PCA transformation ...")
        X_pca = applyPCA(X, numComponents=args.pca)
    print(f"After PCA      : {X_pca.shape}")

    if args.split_type == "disjoint":
        print("\n... Creating Spatially Disjoint Split ...")
        train_patches = []
        train_labels = []
        target_train_patches = []
        target_train_labels = []
        db_patches = []
        db_labels = []
        query_patches = []
        query_labels = []
        # 1. Pad image
        windowSize = args.patch
        margin = (windowSize - 1) // 2
        padded_X = padWithZeros(X_pca, margin)
        
        # Helper function for class-aware split
        def slice_cluster(coords, parts, buffer_size):
            projs = coords[:, 0] + coords[:, 1]
            sorted_idx = np.argsort(projs)
            coords_sorted = coords[sorted_idx]
            projs_sorted = projs[sorted_idx]
            if parts == 2:
                best_split = None
                min_diff = 999999
                for i in range(1, len(coords)-1):
                    train_end_proj = projs_sorted[i-1]
                    test_mask = projs_sorted > (train_end_proj + buffer_size)
                    test_count = np.sum(test_mask)
                    if test_count > 0:
                        diff = abs(i - test_count)
                        if diff < min_diff:
                            min_diff = diff
                            best_split = (i, test_mask)
                if best_split:
                    i, test_mask = best_split
                    return [coords_sorted[:i], coords_sorted[test_mask]]
                return None
            if parts == 3:
                best_split = None
                min_diff = 999999
                for i in range(1, len(coords)-2):
                    p1_end = projs_sorted[i-1]
                    p2_mask = projs_sorted > (p1_end + buffer_size)
                    if np.sum(p2_mask) < 2: continue
                    p2_coords = coords_sorted[p2_mask]
                    p2_projs = projs_sorted[p2_mask]
                    for j in range(1, len(p2_coords)-1):
                        p2_end = p2_projs[j-1]
                        p3_mask = p2_projs > (p2_end + buffer_size)
                        p3_count = np.sum(p3_mask)
                        if p3_count > 0:
                            c1, c2, c3 = i, j, p3_count
                            diff = abs(c1-c2) + abs(c2-c3) + abs(c1-c3)
                            if diff < min_diff:
                                min_diff = diff
                                best_split = (i, p2_mask, j, p3_mask)
                if best_split:
                    i, p2_mask, j, p3_mask = best_split
                    p1 = coords_sorted[:i]
                    p2 = coords_sorted[p2_mask][:j]
                    p3 = coords_sorted[p2_mask][p3_mask]
                    return [p1, p2, p3]
                return None
            return None

        def cluster_and_distribute(coords, buffer_size):
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=buffer_size, min_samples=1, metric='chebyshev').fit_predict(coords)
            unique_labels = np.unique(labels)
            clusters = [coords[labels == l] for l in unique_labels]
            result = [[], [], []]
            if len(clusters) >= 3:
                target_src = len(coords) * args.test_ratio if args.test_ratio < 0.5 else len(coords) * (1.0 - args.test_ratio)
                best_idx = np.argmin([abs(len(c) - target_src) for c in clusters])
                result[0].append(clusters.pop(best_idx))
                target_tt = len(coords) * 0.45
                best_idx = np.argmin([abs(len(c) - target_tt) for c in clusters])
                result[1].append(clusters.pop(best_idx))
                for c in clusters:
                    result[2].append(c)
            elif len(clusters) == 2:
                clusters.sort(key=len)
                result[0].append(clusters[0])
                result[2].append(clusters[1])
            elif len(clusters) == 1:
                slices = slice_cluster(clusters[0], 3, buffer_size)
                if slices:
                    result[0].append(slices[0])
                    result[1].append(slices[1])
                    result[2].append(slices[2])
                else:
                    slices = slice_cluster(clusters[0], 2, buffer_size)
                    if slices:
                        result[0].append(slices[0])
                        result[2].append(slices[1])
                    else:
                        result[0].append(clusters[0])
            for i in range(3):
                if len(result[i]) > 0:
                    result[i] = np.vstack(result[i])
                else:
                    result[i] = np.empty((0, 2), dtype=int)
            return result

        # 2. Collect coordinates for all non-zero labels
        for c in range(1, num_classes + 1):
            coords = np.argwhere(y == c)
            if len(coords) == 0:
                continue
                
            src, tt, ttest = cluster_and_distribute(coords, windowSize)
            
            train_coords = src
            target_train_coords = tt
            target_test_coords = ttest
            
            num_query = int(len(target_test_coords) * args.query_ratio)
            query_coords = target_test_coords[:num_query] if len(target_test_coords) > 0 else []
            db_coords = target_test_coords[num_query:] if len(target_test_coords) > 0 else []
            
            # -----------------------------------------------------------
            # Strict spatial overlap verification
            train_set = set(map(tuple, train_coords)) if len(train_coords) > 0 else set()
            target_train_set = set(map(tuple, target_train_coords)) if len(target_train_coords) > 0 else set()
            target_test_set = set(map(tuple, target_test_coords)) if len(target_test_coords) > 0 else set()
            
            overlap_1 = train_set.intersection(target_train_set)
            overlap_2 = train_set.intersection(target_test_set)
            overlap_3 = target_train_set.intersection(target_test_set)
            
            if len(overlap_1) > 0 or len(overlap_2) > 0 or len(overlap_3) > 0:
                print(f"WARNING: Spatial overlap detected in Class {c}!")
            
            # Extraction helper
            def extract_patches(coord_list):
                p_list = []
                for r, col in coord_list:
                    # r, col are indices in unpadded y
                    # In padded_X, the center is at r+margin, col+margin
                    pr, pc = r + margin, col + margin
                    patch = padded_X[pr - margin : pr + margin + 1, pc - margin : pc + margin + 1]
                    p_list.append(patch)
                return p_list
                
            train_patches.extend(extract_patches(train_coords))
            train_labels.extend([c - 1] * len(train_coords))
            
            target_train_patches.extend(extract_patches(target_train_coords))
            target_train_labels.extend([c - 1] * len(target_train_coords))
            
            db_patches.extend(extract_patches(db_coords))
            db_labels.extend([c - 1] * len(db_coords))
            
            query_patches.extend(extract_patches(query_coords))
            query_labels.extend([c - 1] * len(query_coords))
            
        Xtrain = np.array(train_patches)
        ytrain = np.array(train_labels)
        Xtarget_train = np.array(target_train_patches)
        ytarget_train = np.array(target_train_labels)
        Xdb = np.array(db_patches)
        ydb = np.array(db_labels)
        Xquery = np.array(query_patches)
        yquery = np.array(query_labels)
        
        print(f"Train samples  : {Xtrain.shape[0]}")
        print(f"Target Train   : {Xtarget_train.shape[0]}")
        print(f"Database samples: {Xdb.shape[0]}")
        print(f"Query samples   : {Xquery.shape[0]}")
        
    else:
        # ── Patches ───────────────────────────────────────────────
        print("\n... Creating image cubes ...")
        X_cubes, y_all = createImageCubes(
            X_pca, y, windowSize=args.patch,
        )
        print(f"Cubes shape    : {X_cubes.shape}")

        # ── Train / Test split ────────────────────────────────────
        print("\n... Splitting train / test ...")
        Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(
            X_cubes, y_all, testRatio=args.test_ratio,
        )
        print(f"Train samples  : {Xtrain.shape[0]}")

        # ── Database / Query split ────────────────────────────────
        print(f"\n... Splitting test into Database and Query (query_ratio={args.query_ratio}) ...")
        # Extract query set from test set. 
        # If query_ratio is 1.0, test set acts as both query and database.
        if args.query_ratio < 1.0:
            Xdb, Xquery, ydb, yquery = train_test_split(
                Xtest, ytest, test_size=args.query_ratio, stratify=ytest, random_state=args.seed
            )
        else:
            Xdb, Xquery, ydb, yquery = Xtest, Xtest, ytest, ytest
            

        print(f"Database samples: {Xdb.shape[0]}")
        print(f"Query samples   : {Xquery.shape[0]}")

    # ── Reshape for PyTorch  →  (N, 1, bands, H, W) ──────────
    def to_5d(arr):
        if len(arr) == 0:
            return arr
        # (N, H, W, C) → (N, H, W, C, 1) → (N, 1, C, H, W)
        arr = arr[..., np.newaxis]                     # add channel dim
        arr = arr.transpose(0, 4, 3, 1, 2)             # NCHW-style
        return arr

    Xtrain_5d = to_5d(Xtrain)

    Xdb_5d    = to_5d(Xdb)
    Xquery_5d = to_5d(Xquery)

    print(f"Tensor shape   : {Xtrain_5d.shape}")

    # ── DataLoaders ───────────────────────────────────────────
    train_loader = torch.utils.data.DataLoader(
        HSIDataset(Xtrain_5d, ytrain, augment=args.augment),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    db_loader = torch.utils.data.DataLoader(
        HSIDataset(Xdb_5d, ydb),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    query_loader = torch.utils.data.DataLoader(
        HSIDataset(Xquery_5d, yquery),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, db_loader, query_loader, y, num_classes


# ──────────────────────────────────────────────────────────────
# 6.  TRAINING
# ──────────────────────────────────────────────────────────────

def train(train_loader, db_loader, query_loader, num_classes, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\nDevice: {device}")
    
    import copy
    best_mAP = 0.0
    best_state = None
    eval_epochs = {1, 2, 5, 10, 20, 30, 50, 75, 100}
    
    def evaluate_checkpoint(epoch_num, phase_name):
        nonlocal best_mAP, best_state
        if epoch_num not in eval_epochs:
            return
            
        print(f"\n[Checkpoint] Phase {phase_name}, Epoch {epoch_num}")
        net.eval()
        
        db_codes, db_labels = generate_hash_codes(net, device, db_loader, continuous=False)
        query_codes, query_labels = generate_hash_codes(net, device, query_loader, continuous=False)
        
        db_codes_c, _ = generate_hash_codes(net, device, db_loader, continuous=True)
        query_codes_c, _ = generate_hash_codes(net, device, query_loader, continuous=True)
        
        b_map = calculate_mAP(query_codes, query_labels, db_codes, db_labels)
        c_map = calculate_mAP(query_codes_c, query_labels, db_codes_c, db_labels)
        
        print(f"  Binary mAP     : {b_map*100:.2f}%")
        print(f"  Continuous mAP : {c_map*100:.2f}%")
        
        if b_map > best_mAP:
            best_mAP = b_map
            best_state = copy.deepcopy(net.state_dict())
            print(f"  -> New Best Binary mAP: {b_map*100:.2f}%")
            
        print("")
        net.train()

    # Dynamically extract the actual spectral channel count from the first sample
    sample_data = train_loader.dataset[0][0]
    actual_pca_channels = sample_data.shape[1]

    if args.model == "ssftt":
        net = SSFTTHashNet(
            in_channels=1,
            hash_bit_length=args.hash_bit_length,
            num_tokens=args.num_tokens,
            dim=64,
            use_all_tokens=True,
            num_classes=num_classes,
            pca_channels=actual_pca_channels
        ).to(device)
    elif args.model == "cnn":
        net = CNNBaselineHashNet(hash_bit_length=args.hash_bit_length).to(device)
    elif args.model == "mamba":
        from MambaHashNet import MambaHashNet
        net = MambaHashNet(in_channels=actual_pca_channels, hash_bit_length=args.hash_bit_length, mamba_type='both').to(device)
    elif args.model == "moe_mamba":
        from MoEMambaHashNet import MoEMambaHashNet
        net = MoEMambaHashNet(in_channels=actual_pca_channels, hash_bit_length=args.hash_bit_length).to(device)
    elif args.model == "ssrn":
        from SSRNHashNet import SSRNHashNet
        net = SSRNHashNet(in_channels=1, hash_bit_length=args.hash_bit_length, pca_channels=actual_pca_channels).to(device)
    elif args.model == "a2s2kresnet":
        from A2S2KResNetHashNet import A2S2KResNetHashNet
        net = A2S2KResNetHashNet(in_channels=1, hash_bit_length=args.hash_bit_length, pca_channels=actual_pca_channels).to(device)
    elif args.model == "contextualnet":
        from ContextualHashNet import ContextualHashNet
        net = ContextualHashNet(in_channels=1, hash_bit_length=args.hash_bit_length, pca_channels=actual_pca_channels).to(device)
    elif args.model == "cnn2d":
        from CNN2DHashNet import CNN2DHashNet
        net = CNN2DHashNet(in_channels=actual_pca_channels, hash_bit_length=args.hash_bit_length).to(device)
    elif args.model == "cnn3d":
        from CNN3DHashNet import CNN3DHashNet
        net = CNN3DHashNet(input_channels=actual_pca_channels, hash_bit_length=args.hash_bit_length, patch_size=args.patch).to(device)
    elif args.model == "hybridsn":
        from HybridSNHashNet import HybridSNHashNet
        net = HybridSNHashNet(in_channels=actual_pca_channels, patch_size=args.patch, hash_bit_length=args.hash_bit_length).to(device)
    elif args.model == "morphformer":
        from MorphFormerHashNet import MorphFormerHashNet
        net = MorphFormerHashNet(in_channels=actual_pca_channels, patch_size=args.patch, hash_bit_length=args.hash_bit_length).to(device)
    else:
        raise ValueError("Unknown model")

    num_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters : {num_params}")
    if args.loss_type == "csq":
        criterion = CSQLoss(bit_length=args.hash_bit_length, num_classes=num_classes, lambda_q=args.lambda_q).to(device)
    elif args.loss_type == "dpn":
        criterion = DPNLoss(bit_length=args.hash_bit_length, num_classes=num_classes).to(device)
    elif args.loss_type == "dsh":
        criterion = DSHLoss(num_train=len(train_loader.dataset), bit_length=args.hash_bit_length, num_classes=num_classes, alpha=0.1).to(device)
    elif args.loss_type == "greedyhash":
        criterion = GreedyHashLoss(bit_length=args.hash_bit_length, num_classes=num_classes, alpha=0.1).to(device)
    elif args.loss_type == "hashnet":
        criterion = HashNetLoss(num_train=len(train_loader.dataset), bit_length=args.hash_bit_length, num_classes=num_classes, alpha=0.1, step_continuation=20).to(device)
    elif args.loss_type == "idhn":
        criterion = IDHNLoss(num_train=len(train_loader.dataset), bit_length=args.hash_bit_length, num_classes=num_classes, alpha=0.5, gamma=0.1, lambda_val=0.1).to(device)
    elif args.loss_type == "orthohash":
        criterion = OrthoHashLoss(num_classes=num_classes, bit_length=args.hash_bit_length).to(device)
    elif args.loss_type == "dspch":
        criterion = DSPCHLoss(num_classes=num_classes, bit_length=args.hash_bit_length).to(device)
    elif args.loss_type == "dhnn":
        criterion = BatchDHNNLoss(num_classes=num_classes, bit_length=args.hash_bit_length, margin=2.0).to(device)
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.use_supcon:
        supcon_criterion = SupConLoss(temperature=args.supcon_temp).to(device)
    else:
        supcon_criterion = None

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # --- Pre-Training Diagnostic Check ---
    net.eval()
    print("\n--- Diagnostic Check ---")
    data, target, _, _ = next(iter(train_loader))
    unique_targets, indices = np.unique(target.numpy(), return_index=True)
    diag_data = data[indices].to(device)
    diag_target = target[indices]
    with torch.no_grad():
        logits = net(diag_data)
        print(f"Pre-tanh logits for {len(unique_targets)} classes:")
        for t, logit in zip(diag_target, logits):
            print(f"Class {t.item()}: {logit.cpu().numpy()[:5]}... (first 5 bits)")
    print("------------------------\n")

    total_loss = 0.0
    
    total_steps = args.epochs * len(train_loader)
    current_step = 0

    for epoch in range(args.epochs):
        net.train()
        
        for inputs, labels, _, indices in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if args.lambda_pair > 0:
                logits, pooled = net(inputs, return_features=True)
            else:
                logits = net(inputs)

            if args.loss_type in ["dsh", "greedyhash", "hashnet", "idhn", "orthohash", "dspch", "dhnn"]:
                loss = criterion(logits, labels, ind=indices.to(device), epoch=epoch)
            else:
                loss = criterion(logits, labels)
            
            # Continuous-Binary Pairwise Similarity Preservation
            loss_pair = 0.0
            if args.lambda_pair > 0.0:
                pooled_norm = F.normalize(pooled, p=2, dim=1)
                sim_z = torch.mm(pooled_norm, pooled_norm.t())
                
                hash_norm = F.normalize(torch.tanh(logits), p=2, dim=1)
                sim_h = torch.mm(hash_norm, hash_norm.t())
                
                loss_pair = F.mse_loss(sim_z, sim_h)

            if args.use_supcon:
                s_loss = supcon_criterion(logits, labels)
                batch_loss = loss + args.supcon_weight * s_loss + args.lambda_pair * loss_pair
            else:
                batch_loss = loss + args.lambda_pair * loss_pair

            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            current_step += 1

        avg = total_loss / (epoch + 1)
        
        print(f"[Epoch {epoch+1:3d}/{args.epochs}]  "
              f"loss avg: {avg:.4f}  current: {total_loss:.4f}")
                  
        evaluate_checkpoint(epoch + 1, "1")

    print("Finished Training\n")
    return net, criterion, device, best_state


# ──────────────────────────────────────────────────────────────
# 7.  EVALUATION
# ──────────────────────────────────────────────────────────────

def generate_hash_codes(net, device, data_loader, inject_noise=False, continuous=False):
    net.eval()
    all_codes = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, _, _ in data_loader:
            inputs = inputs.to(device)
            if inject_noise:
                inputs = torch.randn_like(inputs)
            logits = net(inputs)
            
            if continuous:
                # normalize features for cosine similarity
                codes = F.normalize(logits, p=2, dim=1)
            else:
                # Binarize to -1, +1
                codes = torch.sign(logits)
                codes[codes == 0] = 1.0 # resolve 0 to 1
                
            all_codes.append(codes.cpu())
            all_labels.append(labels)
            
    return torch.cat(all_codes, dim=0), torch.cat(all_labels, dim=0)

def calculate_mAP(query_codes, query_labels, db_codes, db_labels):
    num_queries = query_codes.size(0)
    APs = []
    
    # Calculate Similarity (dot product is proportional to Hamming distance for binary codes)
    similarity = torch.matmul(query_codes, db_codes.t())
    
    for i in range(num_queries):
        query_label = query_labels[i]
        sim_row = similarity[i]
        
        # Sort database indices by descending similarity
        _, sorted_indices = torch.sort(sim_row, descending=True)
        sorted_db_labels = db_labels[sorted_indices]
        
        # Binary array of correctness
        correct_matches = (sorted_db_labels == query_label).float()
        total_relevant = correct_matches.sum().item()
        
        if total_relevant == 0:
            APs.append(0.0)
            continue
            
        # Calculate precision at each retrieved item
        cumulative_correct = torch.cumsum(correct_matches, dim=0)
        positions = torch.arange(1, len(correct_matches) + 1, dtype=torch.float32, device=correct_matches.device)
        precisions = cumulative_correct / positions
        
        # Average precision is the mean of precisions at correctly retrieved items
        ap = (precisions * correct_matches).sum().item() / total_relevant
        APs.append(ap)
        
    mAP = sum(APs) / len(APs)
    return mAP

def calculate_pr_curve_vectorized(query_codes, query_labels, db_codes, db_labels, draw_range):
    similarity = torch.matmul(query_codes, db_codes.t())
    _, sorted_indices = torch.sort(similarity, descending=True, dim=1)
    sorted_db_labels = db_labels[sorted_indices]
    query_labels_expanded = query_labels.unsqueeze(1).expand(-1, sorted_db_labels.size(1))
    correct_matches = (sorted_db_labels == query_labels_expanded).float()
    total_relevant = correct_matches.sum(dim=1)
    
    P_list = []
    R_list = []
    for k in draw_range:
        if k > db_codes.size(0):
            break
        retrieved_matches = correct_matches[:, :k].sum(dim=1)
        valid_queries = total_relevant > 0
        if valid_queries.sum() == 0:
            P_list.append(0.0)
            R_list.append(0.0)
            continue
        p = retrieved_matches[valid_queries] / k
        r = retrieved_matches[valid_queries] / total_relevant[valid_queries]
        P_list.append(p.mean().item())
        R_list.append(r.mean().item())
    return P_list, R_list

def evaluate_retrieval(device, net, query_loader, db_loader, inject_noise=False, continuous=False):
    print(f"\nGenerating hash codes for database (noise={inject_noise}, continuous={continuous})...")
    db_codes, db_labels = generate_hash_codes(net, device, db_loader, inject_noise, continuous)
    print(f"Generating hash codes for queries (noise={inject_noise}, continuous={continuous})...")
    query_codes, query_labels = generate_hash_codes(net, device, query_loader, inject_noise, continuous)
    
    print("Calculating mAP...")
    mAP = calculate_mAP(query_codes, query_labels, db_codes, db_labels)
    return mAP, query_codes, query_labels, db_codes, db_labels


# ──────────────────────────────────────────────────────────────
# 8.  MAIN
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generalized HSI classification with SSFTT",
    )
    # ── dataset selection ─────────────────────────────────────
    p.add_argument(
        "--dataset", type=str, default="indian_pines",
        choices=list(DATASETS.keys()) + ["custom"],
        help="Name of the HSI dataset (default: indian_pines)",
    )
    p.add_argument(
        "--data_dir", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        help="Folder containing the .mat files",
    )

    # ── custom dataset overrides (only needed for --dataset custom) ──
    p.add_argument("--data_file",  type=str, default=None)
    p.add_argument("--data_key",   type=str, default=None)
    p.add_argument("--label_file", type=str, default=None)
    p.add_argument("--label_key",  type=str, default=None)
    p.add_argument(
        "--class_names", type=str, default=None,
        help='Comma-separated class names, e.g. "Tree,Water,Road"',
    )

    # ── hyper-parameters ──────────────────────────────────────
    p.add_argument("--pca",        type=int,   default=30)
    p.add_argument("--patch",      type=int,   default=11)
    p.add_argument("--test_ratio", type=float, default=0.90)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=0.001)
    p.add_argument("--hash_bit_length", type=int, default=32)
    p.add_argument("--augment", action="store_true", help="Apply data augmentation")
    p.add_argument("--loss_type", type=str, default="csq", choices=["csq", "dpn", "dsh", "greedyhash", "hashnet", "idhn", "orthohash", "dspch", "dhnn"])
    p.add_argument("--query_ratio", type=float, default=0.1, help="Fraction of test set used as queries")
    p.add_argument("--random_noise_eval", action="store_true", help="Overwrite inputs with random noise during eval")
    p.add_argument("--random_hash_eval", action="store_true", help="Completely ignore network, generate random binary hashes for eval")
    p.add_argument("--shuffle_train_labels", action="store_true", help="Shuffle training labels to verify baseline")
    p.add_argument("--seed", type=int, default=345, help="Random seed for reproducibility")
    p.add_argument("--prepatched_dir", type=str, default=None, help="Path to already patched dataset")
    p.add_argument("--model", type=str, default="ssftt", choices=["ssftt", "cnn", "mamba", "moe_mamba", "ssrn", "a2s2kresnet", "contextualnet", "cnn2d", "cnn3d", "hybridsn", "morphformer"], help="Architecture to run")
    p.add_argument("--split_type", type=str, default="random", choices=["random", "disjoint"], help="How to split train/test")
    p.add_argument("--use_supcon", action="store_true", help="Use Supervised Contrastive Loss")
    p.add_argument("--supcon_weight", type=float, default=0.1, help="Weight for SupCon loss")
    p.add_argument("--supcon_temp", type=float, default=0.1, help="Temperature for SupCon loss")
    p.add_argument("--lambda_q", type=float, default=0.0, help="Quantization penalty weight")
    p.add_argument("--lambda_pair", type=float, default=0.0, help="Continuous-binary pairwise similarity preservation loss weight")
    p.add_argument("--num_tokens", type=int, default=4, help="Number of learned tokens for SSFTT compression")

    # ── output ────────────────────────────────────────────────
    p.add_argument(
        "--output_dir", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"),
    )
    p.add_argument(
        "--run_name", type=str, default=None,
        help="Optional name used to keep this run's weights and report separate",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Handle custom dataset ─────────────────────────────────
    if args.dataset == "custom":
        if not all([args.data_file, args.data_key,
                    args.label_file, args.label_key, args.class_names]):
            raise ValueError(
                "For --dataset custom you must provide: "
                "--data_file --data_key --label_file --label_key --class_names"
            )
        DATASETS["custom"] = {
            "data_file":  args.data_file,
            "data_key":   args.data_key,
            "label_file": args.label_file,
            "label_key":  args.label_key,
            "class_names": [n.strip() for n in args.class_names.split(",")],
        }

    ds_cfg = DATASETS[args.dataset]

    # ── Build data loaders ────────────────────────────────────
    train_loader, db_loader, query_loader, y_all, num_classes = \
        create_data_loader(args)

    # ── Train ─────────────────────────────────────────────────
    tic = time.perf_counter()
    net, criterion, device, best_state = train(train_loader, db_loader, query_loader, num_classes, args)
    
    if best_state is not None:
        net.load_state_dict(best_state)
    train_time = time.perf_counter() - tic

    # ── Save weights ──────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name or (
        f"{args.model}_{args.dataset}_{args.loss_type}_"
        f"pca{args.pca}_patch{args.patch}_bits{args.hash_bit_length}"
    )
    weight_path = os.path.join(
        args.output_dir, f"{run_name}_params.pth",
    )
    torch.save(net.state_dict(), weight_path)
    print(f"Saved weights → {weight_path}")

    # ── Test (Retrieval) ──────────────────────────────────────
    tic = time.perf_counter()
    if args.random_hash_eval:
        print("\n--- RANDOM HASH EVAL ---")
        # generate random query and db codes
        N_q = len(query_loader.dataset)
        N_db = len(db_loader.dataset)
        query_codes = (torch.randint(0, 2, (N_q, args.hash_bit_length)) * 2 - 1).float().to(device)
        db_codes = (torch.randint(0, 2, (N_db, args.hash_bit_length)) * 2 - 1).float().to(device)
        query_labels = query_loader.dataset.y.to(device)
        db_labels = db_loader.dataset.y.to(device)
        mAP = calculate_mAP(query_codes, query_labels, db_codes, db_labels)
        mAP_bin = mAP
        mAP_cont = float("nan")
    else:
        mAP_bin, q_codes_b, q_labels_b, db_codes_b, db_labels_b = evaluate_retrieval(device, net, query_loader, db_loader, inject_noise=args.random_noise_eval, continuous=False)
        mAP_cont, q_codes_c, q_labels_c, db_codes_c, db_labels_c = evaluate_retrieval(device, net, query_loader, db_loader, inject_noise=args.random_noise_eval, continuous=True)
        mAP = mAP_bin  # just to keep downstream variables happy
        
    test_time = time.perf_counter() - tic

    print("\n" + "=" * 60)
    print(f"  Binary mAP     : {mAP_bin * 100:.2f}%")
    print(f"  Continuous mAP : {mAP_cont * 100:.2f}%")
    print("=" * 60 + "\n")

    # ── Write results to file ─────────────────────────────────
    result_path = os.path.join(
        args.output_dir, f"{run_name}_retrieval_report.txt",
    )
    with open(result_path, "w") as f:
        f.write(f"Dataset       : {args.dataset}\n")
        f.write(f"Model         : {args.model}\n")
        f.write(f"Run name      : {run_name}\n")
        f.write(f"PCA components: {args.pca}\n")
        f.write(f"Patch size    : {args.patch}\n")
        f.write(f"Query ratio   : {args.query_ratio}\n")
        f.write(f"Hash bits     : {args.hash_bit_length}\n")
        f.write(f"Loss type     : {args.loss_type}\n")
        f.write(f"Epochs        : {args.epochs}\n")
        f.write(f"LR            : {args.lr}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training time : {train_time:.2f} s\n")
        f.write(f"Eval time     : {test_time:.2f} s\n")
        f.write(f"mAP (%)       : {mAP * 100:.2f}\n")
    print(f"Report saved → {result_path}")

    # ── PR Curve Evaluation ───────────────────────────────────
    if not args.random_hash_eval:
        try:
            from utils.tools import draw_range
        except ImportError:
            draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]

        print("Calculating Precision-Recall Curve (Continuous)...")
        P_list, R_list = calculate_pr_curve_vectorized(q_codes_c, q_labels_c, db_codes_c, db_labels_c, draw_range)
        
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(11, 9))
        plt.plot(R_list, P_list, linestyle="-", marker="D", label=args.model, linewidth=4, markersize=12)
        plt.grid(True)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        pr_path = os.path.join(args.output_dir, f"{run_name}_pr.pdf")
        npz_path = os.path.join(args.output_dir, f"{run_name}_pr.npz")
        np.savez(npz_path, P=P_list, R=R_list)
        plt.savefig(pr_path, bbox_inches='tight')
        print(f"PR curve plot saved → {pr_path}")
        print(f"PR curve data saved → {npz_path}")
        plt.close()