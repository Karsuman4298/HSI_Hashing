import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

class CNN3DHashNet(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, hash_bit_length=32, patch_size=11):
        super().__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (1, 1, 1)

        self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0)
        self.pool1 = nn.Conv3d(20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        self.conv3 = nn.Conv3d(35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, hash_bit_length)
        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels, self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            s0, s1, s2, s3, s4 = x.size()
        return s1 * s2 * s3 * s4

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        pooled = x.view(-1, self.features_size)
        x = self.dropout(pooled)
        hash_codes = self.fc(x)
        
        if return_features:
            return hash_codes, pooled
        return hash_codes
