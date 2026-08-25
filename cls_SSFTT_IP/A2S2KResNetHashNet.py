import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelSELayer3D(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_tensor):
        batch_size, num_channels, D, H, W = input_tensor.size()
        squeeze_tensor = self.avg_pool(input_tensor)
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        return torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

class SpatialSELayer3D(nn.Module):
    def __init__(self, num_channels):
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_tensor, weights=None):
        batch_size, channel, D, H, W = input_tensor.size()
        out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)
        return torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

class ChannelSpatialSELayer3D(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)
    def forward(self, input_tensor):
        return torch.max(self.cSE(input_tensor), self.sSE(input_tensor))

class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w, t = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -3)).transpose(-1, -3).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ResidualA2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1, start_block=False, end_block=False):
        super(ResidualA2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        if not start_block:
            self.bn0 = nn.BatchNorm3d(in_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.ecalayer = eca_layer(out_channels)
        self.start_block = start_block
        self.end_block = end_block

    def forward(self, X):
        identity = X
        if self.start_block:
            out = self.conv1(X)
        else:
            out = self.bn0(X)
            out = F.relu(out)
            out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.start_block:
            out = self.bn2(out)
        out = self.ecalayer(out)
        out += identity
        if self.end_block:
            out = self.bn2(out)
            out = F.relu(out)
        return out

class A2S2KResNetHashNet(nn.Module):
    def __init__(self, in_channels=1, hash_bit_length=64, pca_channels=30, kernel_size_param=24, reduction=2):
        super(A2S2KResNetHashNet, self).__init__()
        band = pca_channels
        self.PARAM_KERNEL_SIZE = kernel_size_param
        self.conv1x1 = nn.Conv3d(in_channels=1, out_channels=self.PARAM_KERNEL_SIZE, kernel_size=(1, 1, 7), stride=(1, 1, 2), padding=0)
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=self.PARAM_KERNEL_SIZE, kernel_size=(3, 3, 7), stride=(1, 1, 2), padding=(1, 1, 0))

        self.batch_norm1x1 = nn.Sequential(nn.BatchNorm3d(self.PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True))
        self.batch_norm3x3 = nn.Sequential(nn.BatchNorm3d(self.PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True))

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_se = nn.Sequential(nn.Conv3d(self.PARAM_KERNEL_SIZE, band // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True))
        self.conv_ex = nn.Conv3d(band // reduction, self.PARAM_KERNEL_SIZE, 1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.res_net1 = ResidualA2(self.PARAM_KERNEL_SIZE, self.PARAM_KERNEL_SIZE, (1, 1, 7), (0, 0, 3), start_block=True)
        self.res_net2 = ResidualA2(self.PARAM_KERNEL_SIZE, self.PARAM_KERNEL_SIZE, (1, 1, 7), (0, 0, 3))
        self.res_net3 = ResidualA2(self.PARAM_KERNEL_SIZE, self.PARAM_KERNEL_SIZE, (3, 3, 1), (1, 1, 0))
        self.res_net4 = ResidualA2(self.PARAM_KERNEL_SIZE, self.PARAM_KERNEL_SIZE, (3, 3, 1), (1, 1, 0), end_block=True)

        kernel_3d = math.ceil((band - 6) / 2)

        self.conv2 = nn.Conv3d(in_channels=self.PARAM_KERNEL_SIZE, out_channels=128, padding=(0, 0, 0), kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=self.PARAM_KERNEL_SIZE, padding=(0, 0, 0), kernel_size=(3, 3, 128), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(nn.BatchNorm3d(self.PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True))

        self.avg_pooling = nn.AdaptiveAvgPool3d(1)
        
        self.hash_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.PARAM_KERNEL_SIZE, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, hash_bit_length)
        )

    def forward(self, X, mask=None, return_features=False):
        # Input shape from dataloader: (B, 1, C, H, W)
        x_in = X.permute(0, 1, 3, 4, 2)
        
        x_1x1 = self.conv1x1(x_in)
        x_1x1 = self.batch_norm1x1(x_1x1).unsqueeze(dim=1)
        x_3x3 = self.conv3x3(x_in)
        x_3x3 = self.batch_norm3x3(x_3x3).unsqueeze(dim=1)

        x1 = torch.cat([x_3x3, x_1x1], dim=1)
        U = torch.sum(x1, dim=1)
        S = self.pool(U)
        Z = self.conv_se(S)
        attention_vector = torch.cat(
            [self.conv_ex(Z).unsqueeze(dim=1), self.conv_ex(Z).unsqueeze(dim=1)],
            dim=1
        )
        attention_vector = self.softmax(attention_vector)
        V = (x1 * attention_vector).sum(dim=1)

        x2 = self.res_net1(V)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        pooled = x4.view(x4.size(0), -1)
        
        hash_codes = self.hash_head(pooled)
        if return_features:
            return hash_codes, pooled
        return hash_codes
