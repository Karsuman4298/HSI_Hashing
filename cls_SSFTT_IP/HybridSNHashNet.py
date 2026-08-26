import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

class HybridSNHashNet(nn.Module):
    def __init__(self, in_channels, patch_size=11, hash_bit_length=32):
        super(HybridSNHashNet, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        
        self.block_1_3D = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.block_2_2D = nn.Sequential(
            nn.Conv2d(32 * (in_channels - 12), 64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True)
        )
        
        self.features_size = self._get_final_flattened_size()

        self.classifier = nn.Sequential(
            nn.Linear(self.features_size, 256),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.Dropout(p=0.4)
        )
        self.hash_head = nn.Linear(128, hash_bit_length)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.in_channels, self.patch_size, self.patch_size))
            y = self.block_1_3D(x)
            y = y.view(-1, y.shape[1] * y.shape[2], y.shape[3], y.shape[4])
            y = self.block_2_2D(y)
            return y.shape[1] * y.shape[2] * y.shape[3]

    def forward(self, x, return_features=False):
        y = self.block_1_3D(x)
        y = y.view(-1, y.shape[1] * y.shape[2], y.shape[3], y.shape[4])
        y = self.block_2_2D(y)
        y = y.view(y.size(0), -1)
        
        pooled = self.classifier(y)
        hash_codes = self.hash_head(pooled)
        
        if return_features:
            return hash_codes, pooled
        return hash_codes
