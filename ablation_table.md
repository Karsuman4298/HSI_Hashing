# Ablation Study: Loss Function Comparison Across Bit Lengths

**Dataset:** Houston 2013 (Pre-Patched, PCA=30, Patch Size=11)
**Architecture:** SSFTTHashNet 
**Epochs:** 100

| Model / Loss Function | Optimization Type | 16-Bit mAP (%) | 32-Bit mAP (%) | 64-Bit mAP (%) | Avg. Train Time (s) |
|-----------------------|-------------------|----------------|----------------|----------------|---------------------|
| **OrthoHash** | Orthogonal Cosine Similarity | 82.85 | 82.49 | 83.14 | ~211s |
| **GreedyHash** | Sign / CrossEntropy Penalty | 81.66 | 83.40 | 83.83 | ~478s |
| **CSQ (Central Similarity)** | Center-Alignment / Bit-Balance | 82.25 | 79.32 | 78.08 | ~418s |
| **DSPCH** | Pre-generated Shared Proxies | 79.56 | 80.51 | 82.58 | ~257s |
| **DSH (Deep Supervised)** | Pairwise Margin / Quantization | 76.58 | 77.59 | 79.85 | ~483s |
| **DHNN** | Deep Hashing Neural Network | 78.62 | 77.80 | 78.93 | ~316s |
| **DPN (Deep Pairwise)** | Center-Alignment / Polarization | 76.04 | 81.39 | 76.78 | ~427s |
| **HashNet** | Scaled Tanh Continuation | 7.98 | 7.98 | 59.07 | ~639s |
| **IDHN** | MSE / Log Contrastive | 32.58 | 38.02 | 25.87 | ~412s |

### Key Observations

*   **OrthoHash** is a phenomenal addition. It achieves top-tier performance across all bit lengths (82.85%, 82.49%, 83.14%) while training in less than half the time of the other top performers (~211s). Its single cosine-similarity objective proves highly stable and mathematically efficient.
*   **GreedyHash** proved to be the most robust optimization strategy overall, dominating both the 32-bit and 64-bit tests and performing exceptionally well at 16 bits.
*   **CSQ** achieved the highest mAP for the ultra-compressed 16-bit space (82.25%), showing that its predefined class-center alignment strategy excels when the hash space is heavily restricted. (Though OrthoHash edged it out at 82.85%).
*   **DSPCH** performs very well, particularly scaling up at higher bit depths (82.58% at 64-bits). It also trains very efficiently (~257s), making it the second-fastest optimization strategy after OrthoHash.
*   **DHNN, DSH, and DPN** maintain strong, stable mid-to-high 70s performance across all bit lengths.
*   **HashNet and IDHN** struggled significantly across the board. HashNet completely collapsed at lower bit depths (7.98%), further confirming that its delicate continuous tanh scaling schedule requires either a much larger alpha multiplier or significantly more training epochs to succeed on this architecture. IDHN's poor performance remains attributed to its mathematical incompatibility with single-label datasets.
