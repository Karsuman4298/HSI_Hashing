import torch
from torch import nn
from einops import rearrange
from SSFTTnet import Transformer

class SSFTTHashNet(nn.Module):
    def __init__(self, in_channels=1, hash_bit_length=32, num_tokens=4, dim=64,
                 depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1,
                 use_all_tokens=True, num_classes=15, pca_channels=30):
        super(SSFTTHashNet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.use_all_tokens = use_all_tokens
        self.num_classes = num_classes

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * (pca_channels - 2), out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        # --- VTS-style hash head ---
        hash_in_features = dim * (num_tokens + 1) if use_all_tokens else dim
        self.hash_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hash_in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, hash_bit_length),
        )

    def forward(self, x, mask=None, return_features=False):
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        
        x = rearrange(x, 'b c h w -> b (h w) c')

        wa = rearrange(self.token_wA, 'b h w -> b w h')
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)

        # --- pool tokens for hashing ---
        if self.use_all_tokens:
            pooled = x.reshape(x.size(0), -1)   # flatten all (num_tokens+1) tokens
        else:
            pooled = x[:, 0]                     # CLS token only

        hash_codes = self.hash_head(pooled)      # continuous logits, pre-tanh/sign
        
        if return_features:
            return hash_codes, pooled
            
        return hash_codes

class CNNBaselineHashNet(nn.Module):
    """
    A simple CNN baseline that uses the exact same 3D/2D convolution layers as SSFTTHashNet
    but skips the entire tokenization and Vision Transformer process. It flattens the conv2d
    output and passes it directly to the hash head.
    """
    def __init__(self, in_channels=1, hash_bit_length=32, dim=64):
        super(CNNBaselineHashNet, self).__init__()
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * 28, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # When patch is 13x13:
        # After 3x3 conv3d without padding, spatial is 11x11
        # After 3x3 conv2d without padding, spatial is 9x9
        # Flattened size: 64 * 9 * 9 = 5184
        flattened_size = 64 * 9 * 9
        
        self.hash_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flattened_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, hash_bit_length),
        )

    def forward(self, x, mask=None):
        # x shape: (N, 1, 30, 13, 13)
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        
        # Flatten directly
        pooled = x.reshape(x.size(0), -1)
        hash_codes = self.hash_head(pooled)
        return hash_codes
