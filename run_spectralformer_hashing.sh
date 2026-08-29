#!/bin/bash

# Example script to train SpectralFormerHashNet for HSI hashing
# Make sure to adjust --dataset, --hash_bit_length, and other arguments as needed.

echo "Training SpectralFormer HashNet on Indian Pines dataset with 16 bits..."
python cls_SSFTT_IP/train_hsi_hashing.py \
    --model spectralformer \
    --dataset indian_pines \
    --hash_bit_length 16 \
    --pca 30 \
    --patch 11 \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 64 \
    --loss_type csq

echo "Training SpectralFormer HashNet on Indian Pines dataset with 32 bits..."
python cls_SSFTT_IP/train_hsi_hashing.py \
    --model spectralformer \
    --dataset indian_pines \
    --hash_bit_length 32 \
    --pca 30 \
    --patch 11 \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 64 \
    --loss_type csq
