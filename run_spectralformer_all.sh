#!/bin/bash

# Datasets to train on
DATASETS=("houston2013" "houston2018" "nilifossae" "trento")

# Hash bit lengths
BITS=(16 32 64)

# Number of epochs
EPOCHS=100

echo "Starting SpectralFormer HashNet training on server..."

for DATASET in "${DATASETS[@]}"; do
    for BIT in "${BITS[@]}"; do
        echo "=========================================================="
        echo "Training SpectralFormer | Dataset: $DATASET | Bits: $BIT"
        echo "=========================================================="
        
        # You can adjust PCA, patch size, or batch size here if needed
        # We redirect output to a log file to track the training process
        LOG_FILE="output_spectralformer_${DATASET}_${BIT}bits.log"
        
        PYTHONPATH=cls_SSFTT_IP python3 cls_SSFTT_IP/train_hsi_hashing.py \
            --model spectralformer \
            --dataset $DATASET \
            --hash_bit_length $BIT \
            --pca 30 \
            --patch 11 \
            --epochs $EPOCHS \
            --lr 0.001 \
            --batch_size 64 \
            --loss_type csq | tee $LOG_FILE
            
        echo "Finished $DATASET with $BIT bits."
        echo ""
    done
done

echo "All trainings completed!"
