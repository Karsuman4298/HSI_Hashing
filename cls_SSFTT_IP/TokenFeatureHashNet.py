"""Explicit CLS/all architecture adaptation for backbones without native CLS tokens."""
import torch
from torch import nn


class TokenFeatureHashNet(nn.Module):
    """Tokenize the final unpooled feature map and add one transformer block.

    Every spatial/spectral position becomes a token, projected to width 64.
    Both variants share this architecture up to the hash head; CLS uses the
    first encoded token and all concatenates CLS and every feature token.
    """
    def __init__(self, backbone, example, bits, feature_mode, dim=64):
        super().__init__()
        if feature_mode not in ('cls', 'all'):
            raise ValueError('feature_mode must be cls or all')
        self.backbone = backbone
        self.feature_mode = feature_mode
        training = backbone.training
        backbone.eval()
        with torch.no_grad():
            features = backbone(example, return_tokens=True)
        backbone.train(training)
        if features.ndim not in (4, 5):
            raise ValueError('Expected an unpooled channel-first 2D/3D feature map')
        channels = features.shape[1]
        count = features.flatten(2).shape[2]
        # The native pooling/classification heads are bypassed by return_tokens.
        for name in ('hash_head', 'classifier', 'fc', 'fc_1', 'fc_2'):
            if hasattr(backbone, name):
                setattr(backbone, name, nn.Identity())
        self.projection = nn.Linear(channels, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.empty(1, count + 1, dim))
        nn.init.normal_(self.position, std=0.02)
        self.encoder = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=128,
                                                   dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        width = dim if feature_mode == 'cls' else dim * (count + 1)
        self.hash_head = nn.Sequential(nn.Dropout(0.5), nn.Linear(width, 1024),
                                       nn.ReLU(inplace=True), nn.Linear(1024, bits))

    def forward(self, x, return_features=False):
        features = self.backbone(x, return_tokens=True)
        tokens = self.projection(features.flatten(2).transpose(1, 2))
        tokens = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), tokens], dim=1)
        tokens = self.norm(self.encoder(tokens + self.position))
        pooled = tokens[:, 0] if self.feature_mode == 'cls' else tokens.flatten(1)
        codes = self.hash_head(pooled)
        return (codes, pooled) if return_features else codes
