import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ContextualHashNet(nn.Module):
    """
    Adapted from CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def __init__(self, in_channels=1, hash_bit_length=64, pca_channels=30):
        super(ContextualHashNet, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 128, (3, 3, pca_channels), stride=(1, 1, 2), padding=(1, 1, 0))
        self.conv_1x1 = nn.Conv3d(
            1, 128, (1, 1, pca_channels), stride=(1, 1, 1), padding=0)

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(256, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        
        # Adaptive pooling to support any patch size
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.hash_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, hash_bit_length)
        )

        self.apply(self.weight_init)

    def forward(self, X, mask=None, return_features=False):
        # Input shape from dataloader: (B, 1, C, H, W)
        # Convert to LeeEtAl expected shape: (B, 1, H, W, BAND)
        x_in = X.permute(0, 1, 3, 4, 2)

        # Inception module
        x_3x3 = self.conv_3x3(x_in)
        x_1x1 = self.conv_1x1(x_in)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        
        # Safely remove the last dimension (BAND dimension which is now 1)
        x = x.squeeze(-1)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        
        # Pool spatial dimensions to 1x1
        x = self.avg_pooling(x)
        pooled = x.view(x.size(0), -1)
        
        hash_codes = self.hash_head(pooled)
        if return_features:
            return hash_codes, pooled
        return hash_codes
