import sys
import numpy as np
import argparse
from cls_SSFTT_IP.train_hsi_hashing import create_data_loader

def check_dataset(dataset_name):
    class DummyArgs:
        def __init__(self, ds):
            self.dataset = ds
            self.data_dir = "data"
            self.pca = 30
            self.patch = 13
            self.split_type = "disjoint"
            self.test_ratio = 0.9
            self.query_ratio = 0.1
            self.seed = 42
            self.batch_size = 64
            self.augment = False
            self.loss_type = "csq"
            self.hash_bit_length = 32
            
    args = DummyArgs(dataset_name)
    
    print(f"\n==================================================")
    print(f"DIAGNOSTIC FOR {dataset_name.upper()}")
    print(f"==================================================")
    
    try:
        loaders = create_data_loader(args)
    except Exception as e:
        print(f"Failed to load: {e}")
        
if __name__ == "__main__":
    check_dataset("houston2013")
    check_dataset("indian_pines")
