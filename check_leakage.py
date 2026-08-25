import h5py
import numpy as np

print("Loading Train patches...")
with h5py.File('cls_SSFTT_IP/NiliFossae_dataset/HSI_Tr.mat', 'r') as f:
    X_train = np.array(f['NiliFossae_Tr'])  # (5339, 11, 11, 30)
    
print("Loading Test patches...")
with h5py.File('cls_SSFTT_IP/NiliFossae_dataset/HSI_Te.mat', 'r') as f:
    X_test = np.array(f['NiliFossae_Te'])  # (21371, 11, 11, 30)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Flatten Train patches to get all unique pixels seen during training
# Shape: (5339 * 11 * 11, 30)
train_pixels = X_train.reshape(-1, X_train.shape[-1])

# Extract only the CENTER pixel of the Test patches
# Shape: (21371, 30)
center_idx = X_test.shape[1] // 2
test_center_pixels = X_test[:, center_idx, center_idx, :]

print("Building fast hash set for training pixels...")
# Convert each 30-D pixel to a tuple for fast hash set lookup
# We round to 4 decimal places to avoid floating point precision issues in MATLAB/HDF5 conversions
train_set = set([tuple(np.round(p, 4)) for p in train_pixels])

print("Checking for spatial data leakage (Test centers inside Train patches)...")
leakage_count = 0
for p in test_center_pixels:
    if tuple(np.round(p, 4)) in train_set:
        leakage_count += 1

leakage_percentage = (leakage_count / len(test_center_pixels)) * 100
print(f"\n--- LEAKAGE REPORT ---")
print(f"Total Test Patches: {len(test_center_pixels)}")
print(f"Test Patches with center pixel ALREADY SEEN in Training Patches: {leakage_count}")
print(f"Data Leakage Percentage: {leakage_percentage:.2f}%")

if leakage_percentage > 10:
    print("\nCONCLUSION: NOT DISJOINT! You have massive spatial data leakage. This is why you are getting 98%!")
else:
    print("\nCONCLUSION: DISJOINT! The dataset looks properly spatially separated.")
