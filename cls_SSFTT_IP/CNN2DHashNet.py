import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

class CNN2DHashNet(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, hash_bit_length=32):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, 256, 3)
        self.conv_2 = nn.Conv2d(256, 512, 3)
        self.mp = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc_1 = nn.Linear(512, 128)
        self.fc_2 = nn.Linear(128, hash_bit_length)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, return_features=False):
        # x shape: (B, 1, C, H, W)
        x = x.squeeze(dim=1)
        x = self.conv_1(x)
        x = self.mp(x)
        x = self.relu(x)
        x = self.conv_2(x)
        
        # Use adaptive pool to handle variable patch sizes
        x = self.adaptive_pool(x)
        x = x.view(-1, x.shape[1])
        
        pooled = self.relu(self.fc_1(x))
        hash_codes = self.fc_2(pooled)
        
        if return_features:
            return hash_codes, pooled
        return hash_codes
