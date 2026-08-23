# Ablation Study Results

### Dataset: Houston2013
**Backbone:** SSFTT (PCA: 30, Patch: 11, Epochs: 100, LR: 0.001)

| Loss Type | Hash Bits | mAP (%) | Training Time (s) | Eval Time (s) |
|-----------|-----------|---------|-------------------|---------------|
| **dspch** | 64        | 82.58   | 235.19            | 8.82          |
| **dhnn**  | 64        | 78.93   | 311.84            | 8.83          |
| **dspch** | 32        | 80.51   | 281.83            | 11.16         |
| **dhnn**  | 32        | 77.80   | 442.42            | 12.70         |
| **dspch** | 16        | 79.56   | 358.88            | 8.70          |
| **dhnn**  | 16        | 78.62   | 193.89            | 4.51          |

*(Note: These results represent the SSFTT baseline using the newly added DSPCH and DHNN loss functions.)*
