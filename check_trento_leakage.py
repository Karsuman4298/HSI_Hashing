import h5py
import numpy as np
import scipy.io as sio

def get_data(file_path):
    try:
        # Try scipy first
        mat = sio.loadmat(file_path)
        valid_keys = [k for k in mat.keys() if not k.startswith('__') and 'label' not in k.lower() and 'gt' not in k.lower()]
        key = valid_keys[0] if valid_keys else list(mat.keys())[-1]
        data = np.array(mat[key])
    except:
        # Fallback to h5py
        with h5py.File(file_path, 'r') as f:
            valid_keys = [k for k in f.keys() if not k.startswith('#') and 'label' not in k.lower() and 'gt' not in k.lower()]
            key = valid_keys[0] if valid_keys else list(f.keys())[-1]
            data = np.array(f[key])
            if len(data.shape) == 4 and data.shape[0] < data.shape[-1]:
                data = data.T
            elif len(data.shape) == 3 and data.shape[0] < data.shape[-1]:
                data = data.T
    return data

print("Loading Trento Train patches...")
X_train = get_data('/scratch/skaushik8/HSI_Hashing/Trento/HSI_Tr.mat')
    
print("Loading Trento Test patches...")
X_test = get_data('/scratch/skaushik8/HSI_Hashing/Trento/HSI_Te.mat')

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

train_pixels = X_train.reshape(-1, X_train.shape[-1])
center_idx = X_test.shape[1] // 2
test_center_pixels = X_test[:, center_idx, center_idx, :]

print("Building fast hash set for training pixels...")
train_set = set([tuple(np.round(p, 4)) for p in train_pixels])

print("Checking for spatial data leakage (Test centers inside Train patches)...")
leakage_count = 0
for p in test_center_pixels:
    if tuple(np.round(p, 4)) in train_set:
        leakage_count += 1

leakage_percentage = (leakage_count / len(test_center_pixels)) * 100
print(f"\n--- LEAKAGE REPORT FOR TRENTO ---")
print(f"Total Test Patches: {len(test_center_pixels)}")
print(f"Test Patches with center pixel ALREADY SEEN in Training Patches: {leakage_count}")
print(f"Data Leakage Percentage: {leakage_percentage:.2f}%")
