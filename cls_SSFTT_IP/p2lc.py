import numpy as np
import torch

def p2lc_greedy_assignment(P, d_prior, U=0.05):
    """
    P2LC Greedy Assignment Algorithm with Uncertainty Bounds.
    P: (N, C) probability matrix (model outputs on target-train).
    d_prior: (C,) target marginal class distribution prior (e.g. from source).
    U: uncertainty coefficient controlling allowed deviation from d_prior.
    
    Returns:
        corrected_labels: (N,) hard assignments
        confidences: (N,) confidence scores
        d_p: (C,) raw target argmax distribution (for logging)
    """
    P = P.cpu().numpy() if torch.is_tensor(P) else P
    N, C = P.shape
    
    # 1. Estimate target pseudo-class distribution from raw argmax
    raw_preds = P.argmax(axis=1)
    d_p = np.zeros(C)
    for c in range(C):
        d_p[c] = (raw_preds == c).sum() / N
        
    # 2. Calculate Upper Bounds for each class based on the PRIOR
    # The bounds allow the assignments to deviate by + U from d_prior
    UB = np.ceil(N * np.clip(d_prior + U, 0.0, 1.0)).astype(int)
    
    # 3. Greedy Assignment
    # Sort all (sample, class) pairs by probability descending
    # To do this efficiently, we can flatten the matrix and argsort
    P_flat = P.flatten()
    sorted_indices = np.argsort(-P_flat)
    
    corrected_labels = np.full(N, -1, dtype=int)
    class_counts = np.zeros(C, dtype=int)
    
    for idx in sorted_indices:
        sample_idx = idx // C
        class_idx = idx % C
        
        # If this sample is already assigned, skip
        if corrected_labels[sample_idx] != -1:
            continue
            
        # If assigning to this class violates the upper bound, skip
        if class_counts[class_idx] >= UB[class_idx]:
            continue
            
        # Assign
        corrected_labels[sample_idx] = class_idx
        class_counts[class_idx] += 1
        
    # Edge case: If any samples are completely unassigned (because all their non-zero
    # probability classes hit UB), we forcefully assign them to their max available class.
    unassigned = np.where(corrected_labels == -1)[0]
    for u in unassigned:
        # Find the class with max prob that hasn't hit UB
        sorted_classes = np.argsort(-P[u])
        for c in sorted_classes:
            if class_counts[c] < UB[c]:
                corrected_labels[u] = c
                class_counts[c] += 1
                break
        
        # If ALL classes somehow hit UB (mathematically impossible if sum(UB) >= N, but just in case)
        if corrected_labels[u] == -1:
            best_c = np.argmax(P[u])
            corrected_labels[u] = best_c
            class_counts[best_c] += 1
            
    confidences = P[np.arange(N), corrected_labels]
    
    return torch.tensor(corrected_labels), torch.tensor(confidences), d_p

def analyze_correction(raw_preds, corrected_labels, d_p, N, C):
    if torch.is_tensor(raw_preds):
        raw_preds = raw_preds.cpu().numpy()
    if torch.is_tensor(corrected_labels):
        corrected_labels = corrected_labels.cpu().numpy()
    
    # Percentage of labels changed
    changed_mask = (raw_preds != corrected_labels)
    pct_changed = changed_mask.sum() / N * 100.0
    
    # Target class distribution after correction
    d_p_after = np.zeros(C)
    for c in range(C):
        d_p_after[c] = (corrected_labels == c).sum() / N
        
    return pct_changed, d_p_after
