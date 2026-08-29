#!/bin/bash

echo "Starting Comprehensive Study for SpectralFormer on the server..."

# Adjust the datasets, losses, and output directory as needed
python3 run_comprehensive_study.py \
    --datasets houston2013 houston2018 nilifossae trento \
    --models spectralformer \
    --losses csq dpn dsh greedyhash hashnet idhn orthohash dspch dhnn \
    --bits 16 32 64 \
    --epochs 100 \
    --patch 11 \
    --pca 30 \
    --output_dir output_spectralformer_study

echo "Comprehensive Study completed. Check the output_spectralformer_study directory for results."
