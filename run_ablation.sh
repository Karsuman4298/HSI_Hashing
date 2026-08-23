#!/bin/bash
cd /Users/sumankar/Desktop/HSI_SSFTT
echo "LossType,Binary_mAP" > ablation_results.csv
for loss in csq dpn dsh greedyhash hashnet idhn; do
    echo "Running 100 epochs for $loss..."
    PYTHONPATH=cls_SSFTT_IP python3 cls_SSFTT_IP/train_hsi_hashing.py --dataset houston2013 --prepatched_dir cls_SSFTT_IP/houston13_alreadypatched_dataset --hash_bit_length 64 --epochs 100 --pca 30 --loss_type $loss > output_log_${loss}.txt
    
    b_map=$(grep "mAP (%)" output/retrieval_report_houston2013.txt | awk '{print $4}')
    
    echo "$loss,$b_map" >> ablation_results.csv
    echo "Finished $loss. mAP: $b_map%"
done
