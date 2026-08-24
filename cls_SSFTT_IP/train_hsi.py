"""
Generalized HSI Classification Training Script (SSFTT)
=======================================================
Works with: Indian Pines, Pavia University, Pavia Center, Salinas,
            Houston 2013, Houston 2018, Kennedy Space Center, Botswana,
            WHU-Hi LongKou, and any custom HSI .mat pair.

Usage:
    python train_hsi.py --dataset pavia_university
    python train_hsi.py --dataset indian_pines --pca 30 --patch 13 --test_ratio 0.9
    python train_hsi.py --dataset custom \
        --data_file MyData.mat --data_key my_data \
        --label_file MyGT.mat  --label_key my_gt \
        --class_names "Tree,Water,Road"
"""

import os
import argparse
import time
import numpy as np
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
import torch.optim as optim

import seaborn as sns
import matplotlib.pyplot as plt

import SSFTTnet
import get_cls_map  # your existing visualization module


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
        "label_key":  "paviaU_gt",
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
        "label_key":  "salinas_gt",
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
        "data_file":  "Houston13.mat",
        "data_key":   "Houston",
        "label_file": "Houston13_7gt.mat",
        "label_key":  "Houston_gt",
        "class_names": [
            "Healthy-grass", "Stressed-grass", "Synthetic-grass",
            "Trees", "Soil", "Water", "Residential", "Commercial",
            "Road", "Highway", "Railway", "Parking-lot-1",
            "Parking-lot-2", "Tennis-court", "Running-track",
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
            "Corn", "Cotton", "Sesame", "Broad-leaf-soybean",
            "Narrow-leaf-soybean", "Rice", "Water",
            "Roads-and-houses", "Mixed-weed",
        ],
    },
}


# ──────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ──────────────────────────────────────────────────────────────

import h5py
import scipy.io as sio
import numpy as np

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


def load_single_mat(file_path, preferred_key=None):
    """
    Robust reader: Automatically detects whether the file is standard 
    MATLAB format or MATLAB v7.3 (HDF5).
    """
    try:
        # 1. Try standard SciPy loadmat (MATLAB <= v7)
        mat_dict = sio.loadmat(file_path)
        key = _find_mat_key(mat_dict, preferred_key)
        return np.array(mat_dict[key])
    except (NotImplementedError, Exception):
        # 2. Fallback to h5py for MATLAB v7.3 files
        with h5py.File(file_path, 'r') as f:
            if preferred_key and preferred_key in f:
                data = np.array(f[preferred_key])
            else:
                # Find the first key that isn't metadata
                valid_keys = [k for k in f.keys() if not k.startswith('#')]
                if not valid_keys:
                    raise KeyError(f"No valid dataset found in {file_path}")
                data = np.array(f[valid_keys[0]])
            
            # MATLAB v7.3 saves arrays in reverse dimension order (C-order vs Fortran-order)
            # Transposing (.T) restores the original (Height, Width, Bands) shape
            data = data.T
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

    def __init__(self, X, y):
        self.x = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


# ──────────────────────────────────────────────────────────────
# 5.  DATA-LOADER FACTORY
# ──────────────────────────────────────────────────────────────

def create_data_loader(args):
    """End-to-end: raw .mat → train / test / all DataLoaders."""

    ds_cfg = DATASETS[args.dataset]
    num_classes = len(ds_cfg["class_names"])

    # ── Load ──────────────────────────────────────────────────
    X, y = loadData(args.data_dir, ds_cfg)
    print(f"Dataset        : {args.dataset}")
    print(f"HSI shape      : {X.shape}")
    print(f"Label shape    : {y.shape}")
    print(f"Num classes    : {num_classes}")

    # ── PCA ───────────────────────────────────────────────────
    print("\n... PCA transformation ...")
    X_pca = applyPCA(X, numComponents=args.pca)
    print(f"After PCA      : {X_pca.shape}")

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
    print(f"Test  samples  : {Xtest.shape[0]}")

    # ── Reshape for PyTorch  →  (N, 1, bands, H, W) ──────────
    def to_5d(arr):
        # (N, H, W, C) → (N, H, W, C, 1) → (N, 1, C, H, W)
        arr = arr[..., np.newaxis]                     # add channel dim
        arr = arr.transpose(0, 4, 3, 1, 2)             # NCHW-style
        return arr

    Xtrain_5d = to_5d(Xtrain)
    Xtest_5d  = to_5d(Xtest)
    Xall_5d   = to_5d(X_cubes)

    print(f"Tensor shape   : {Xtrain_5d.shape}")

    # ── DataLoaders ───────────────────────────────────────────
    train_loader = torch.utils.data.DataLoader(
        HSIDataset(Xtrain_5d, ytrain),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        HSIDataset(Xtest_5d, ytest),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    all_loader = torch.utils.data.DataLoader(
        HSIDataset(Xall_5d, y_all),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    return train_loader, test_loader, all_loader, y, num_classes


# ──────────────────────────────────────────────────────────────
# 6.  TRAINING
# ──────────────────────────────────────────────────────────────

def train(train_loader, num_classes, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\nDevice: {device}")

    net = SSFTTnet.SSFTTnet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    total_loss = 0.0
    for epoch in range(args.epochs):
        net.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / (epoch + 1)
        print(f"[Epoch {epoch+1:3d}/{args.epochs}]  "
              f"loss avg: {avg:.4f}  current: {loss.item():.4f}")

    print("Finished Training\n")
    return net, device


# ──────────────────────────────────────────────────────────────
# 7.  EVALUATION
# ──────────────────────────────────────────────────────────────

def test(device, net, test_loader):
    net.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            y_pred.append(preds)
            y_true.append(labels.numpy())
    return np.concatenate(y_pred), np.concatenate(y_true)


def AA_andEachClassAccuracy(cm):
    diag = np.diag(cm)
    row_sum = np.sum(cm, axis=1)
    # prevent division by zero for classes with 0 instances
    with np.errstate(divide='ignore', invalid='ignore'):
        each_acc = np.nan_to_num(truediv(diag, row_sum))
    return each_acc, np.mean(each_acc)

def plot_confusion_matrix(cm, y_true, y_pred, class_names, save_path):
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    actual_class_names = [class_names[i] for i in unique_classes]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=actual_class_names,
                yticklabels=actual_class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def acc_reports(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(cm)
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    actual_class_names = [class_names[i] for i in unique_classes]

    report = classification_report(
        y_true, y_pred, digits=4, 
        labels=unique_classes,
        target_names=actual_class_names,
        zero_division=0
    )
    return report, oa * 100, cm, each_acc * 100, aa * 100, kappa * 100


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
    p.add_argument("--patch",      type=int,   default=13)
    p.add_argument("--test_ratio", type=float, default=0.90)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=0.001)

    # ── output ────────────────────────────────────────────────
    p.add_argument(
        "--output_dir", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"),
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

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
    train_loader, test_loader, all_loader, y_all, num_classes = \
        create_data_loader(args)

    # ── Train ─────────────────────────────────────────────────
    tic = time.perf_counter()
    net, device = train(train_loader, num_classes, args)
    train_time = time.perf_counter() - tic

    # ── Save weights ──────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    config_str = f"{args.dataset}_pca{args.pca}_patch{args.patch}_tr{args.test_ratio}_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}"
    weight_path = os.path.join(
        args.output_dir, f"SSFTT_{config_str}_params.pth",
    )
    torch.save(net.state_dict(), weight_path)
    print(f"Saved weights → {weight_path}")

    # ── Test ──────────────────────────────────────────────────
    tic = time.perf_counter()
    y_pred, y_true = test(device, net, test_loader)
    test_time = time.perf_counter() - tic

    # ── Metrics ───────────────────────────────────────────────
    report, oa, cm, each_acc, aa, kappa = acc_reports(
        y_true, y_pred, ds_cfg["class_names"],
    )

    print("\n" + "=" * 60)
    print(f"  OA    : {oa:.2f}%")
    print(f"  AA    : {aa:.2f}%")
    print(f"  Kappa : {kappa:.2f}%")
    print("=" * 60 + "\n")

    # ── Write results to file ─────────────────────────────────
    result_path = os.path.join(
        args.output_dir, f"classification_report_{config_str}.txt",
    )
    with open(result_path, "w") as f:
        f.write(f"Dataset       : {args.dataset}\n")
        f.write(f"PCA components: {args.pca}\n")
        f.write(f"Patch size    : {args.patch}\n")
        f.write(f"Test ratio    : {args.test_ratio}\n")
        f.write(f"Epochs        : {args.epochs}\n")
        f.write(f"LR            : {args.lr}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training time : {train_time:.2f} s\n")
        f.write(f"Test time     : {test_time:.2f} s\n")
        f.write(f"Kappa  (%)    : {kappa:.2f}\n")
        f.write(f"OA     (%)    : {oa:.2f}\n")
        f.write(f"AA     (%)    : {aa:.2f}\n")
        f.write(f"Each acc (%)  : {each_acc}\n\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    print(f"Report saved → {result_path}")

    # ── Confusion Matrix Heatmap (Seaborn) ────────────────────
    cm_path = os.path.join(args.output_dir, f"confusion_matrix_{config_str}.png")
    plot_confusion_matrix(cm, y_true, y_pred, ds_cfg["class_names"], cm_path)
    print(f"Confusion matrix saved → {cm_path}")

    # ── Classification map ────────────────────────────────────
    get_cls_map.get_cls_map(net, device, all_loader, y_all, dataset_name=config_str)