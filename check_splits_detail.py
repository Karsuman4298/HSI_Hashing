import sys
import numpy as np
from cls_SSFTT_IP.train_hsi_hashing import loadData, DATASETS, applyPCA, padWithZeros

def check_split(dataset_name):
    print(f"\n==================================================")
    print(f"DIAGNOSTIC FOR {dataset_name.upper()}")
    print(f"==================================================")
    
    ds_cfg = DATASETS[dataset_name]
    num_classes = len(ds_cfg["class_names"])
    
    X, y = loadData("data", ds_cfg)
    
    orig_bands = X.shape[2]
    
    X_pca = applyPCA(X, numComponents=30)
    print(f"- original number of bands: {orig_bands}")
    print(f"- bands after PCA: {X_pca.shape[2]}")
    print(f"- image dimensions: {X.shape[0]}x{X.shape[1]}")
    
    active_classes = 0
    for c in range(1, num_classes + 1):
        if np.sum(y == c) > 0:
            active_classes += 1
            
    print(f"- number of active classes: {active_classes}")
    
    print("\n| Class | Name | Source Train | Target Train | Target Test | Query | Database |")
    print("|-------|------|--------------|--------------|-------------|-------|----------|")
    
    total_src = 0
    total_ttrain = 0
    total_ttest = 0
    total_query = 0
    total_db = 0
    
    buffer_size = 13
    
    for c in range(1, num_classes + 1):
        coords = np.argwhere(y == c)
        if len(coords) == 0:
            print(f"| {c:2d} | {ds_cfg['class_names'][c-1]:<20} | 0 | 0 | 0 | 0 | 0 |")
            continue
            
        projs = coords[:, 0] + coords[:, 1]
        sorted_idx = np.argsort(projs)
        coords_sorted = coords[sorted_idx]
        projs_sorted = projs[sorted_idx]
        
        # Source Train
        num_train = int(len(coords) * (1.0 - 0.9))
        train_coords = coords_sorted[:num_train]
        
        max_train_proj = projs_sorted[num_train - 1] if num_train > 0 else -9999
        
        test_mask = projs_sorted > (max_train_proj + buffer_size)
        test_candidates = coords_sorted[test_mask]
        test_projs = projs_sorted[test_mask]
        
        # Target Train
        num_target_train = int(len(test_candidates) * 0.5)
        target_train_coords = test_candidates[:num_target_train]
        
        max_target_train_proj = test_projs[num_target_train - 1] if num_target_train > 0 else -9999
        target_test_mask = test_projs > (max_target_train_proj + buffer_size)
        
        # Target Test
        target_test_coords = test_candidates[target_test_mask]
        
        num_query = int(len(target_test_coords) * 0.1)
        query_coords = target_test_coords[:num_query]
        db_coords = target_test_coords[num_query:]
        
        print(f"| {c:2d} | {ds_cfg['class_names'][c-1]:<20} | {len(train_coords):<12} | {len(target_train_coords):<12} | {len(target_test_coords):<11} | {len(query_coords):<5} | {len(db_coords):<8} |")
        
        total_src += len(train_coords)
        total_ttrain += len(target_train_coords)
        total_ttest += len(target_test_coords)
        total_query += len(query_coords)
        total_db += len(db_coords)
        
    print(f"|    | TOTAL | {total_src:<12} | {total_ttrain:<12} | {total_ttest:<11} | {total_query:<5} | {total_db:<8} |\n")
    

if __name__ == "__main__":
    check_split("houston2013")
    check_split("indian_pines")
