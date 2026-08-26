import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Dropout
from einops import rearrange
import math
import numpy as np
import copy

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

FM = 16

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class Morphology(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        x = fixed_padding(x, self.kernel_size, dilation=1)
        x = self.unfold(x) 
        x = x.unsqueeze(1) 
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))
        weight = self.weight.view(self.out_channels, -1)
        weight = weight.unsqueeze(0).unsqueeze(-1) 
        if   self.type == 'erosion2d':  x = weight - x 
        elif self.type == 'dilation2d': x = weight + x 
        else:
            raise ValueError
        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False) 
        else:
            x = torch.logsumexp(x*self.beta, dim=2, keepdim=False) / self.beta 
        if self.type == 'erosion2d': x = -1 * x
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  
        return x

class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')

class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')

class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding = None, bias = None,p = 64, g = 64):
        super(HetConv, self).__init__()
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,groups=g,padding = kernel_size//3, stride = stride)
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1,groups=p, stride = stride)
    def forward(self, x):
        return self.gwc(x) + self.pwc(x)

class SpectralMorph(nn.Module):
    def __init__(self,FM,NC, kernel = 3):
        super(SpectralMorph, self).__init__()
        self.erosion = Erosion2d(NC, FM, kernel, soft_max=False)
        self.conv1 = nn.Conv2d(FM,FM,1,padding = 0)
        self.dilation = Dilation2d(NC, FM, kernel, soft_max=False)
        self.conv2 = nn.Conv2d(FM,FM,1,padding = 0)
    def forward(self, x):
        z1 = self.erosion(x)
        z1 = self.conv1(z1)
        z2 = self.dilation(x)
        z2 = self.conv2(z2)
        return z1 + z2

class SpatialMorph(nn.Module):
    def __init__(self,FM,NC, kernel = 3):
        super(SpatialMorph, self).__init__()
        self.erosion = Erosion2d(NC, FM, kernel, soft_max=False)
        self.conv1 = nn.Conv2d(FM,FM,3,padding = 1)
        self.dilation = Dilation2d(NC, FM, kernel, soft_max=False)
        self.conv2 = nn.Conv2d(FM,FM,3,padding = 1)
    def forward(self, x):
        z1 = self.erosion(x)
        z1 = self.conv1(z1)
        z2 = self.dilation(x)
        z2 = self.conv2(z2)
        return z1 + z2

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, blockNum = 0):
        super(Block, self).__init__()
        self.hidden_size = dim
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        kernels = [3,5]
        self.cls_norm = LayerNorm(dim, eps=1e-6)
        self.spec_morph = nn.Sequential(SpectralMorph(FM,FM*2,kernels[blockNum]),nn.BatchNorm2d(FM),nn.GELU())
        self.spat_morph = nn.Sequential(SpatialMorph(FM,FM*2,kernels[blockNum]),nn.BatchNorm2d(FM),nn.GELU())
        self.attn = CrossAttention(dim)
    def forward(self, x):
        ht,w = x.shape[2:]
        rest = x[:,1:]
        rest1 = rest
        rest1 = self.spec_morph(rest1)
        rest2 = rest
        rest2 = self.spat_morph(rest2)
        rest = torch.cat([rest1,rest2],dim = 1)
        x = torch.cat([x[:,0:1,:],rest],dim = 1)
        clsTok = x[:,0:1]
        h = clsTok
        clsTok= self.attn(self.attention_norm(x.reshape(x.shape[0],x.shape[1],-1))).reshape(x.shape[0],1,ht,w)
        clsTok = clsTok + h
        clsTok = self.cls_norm(clsTok.reshape(clsTok.shape[0],clsTok.shape[1],-1)).reshape(clsTok.shape)
        x = torch.cat([clsTok,x[:,1:]],dim = 1)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads= 8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for i in range(2):
            layer = Block(dim,i)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        for layer_block in self.layer:
            x = layer_block(x)
        x= x.reshape(x.shape[0],x.shape[1],-1)
        x = self.encoder_norm(x)
        return x[:,0]

class MorphFormerHashNet(nn.Module):
    def __init__(self, in_channels, patch_size=11, hash_bit_length=32, FM=16):
        super(MorphFormerHashNet, self).__init__()
        self.patchsize = patch_size
        NC = in_channels
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0,1,1), stride = 1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            HetConv(8 * (NC - 8), FM*4,
                p = 1,
                g = (FM*4)//4 if (8 * (NC - 8))%FM == 0 else (FM*4)//8,
                   ),
            nn.BatchNorm2d(FM*4),
            nn.ReLU()
        )
        self.ca = CrossAttentionBlock(FM*4)
        
        self.hash_head = nn.Linear(FM*4, hash_bit_length)
        torch.nn.init.xavier_uniform_(self.hash_head.weight)
        torch.nn.init.normal_(self.hash_head.bias, std=1e-6)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, FM*4))
        self.position_embeddings = nn.Parameter(torch.zeros(1, FM*2 + 1, FM*4))
        self.dropout = nn.Dropout(0.1)
        self.FM = FM
        self.token_wA = nn.Parameter(torch.empty(1, FM*2, 64), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, 64), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wV)

    def forward(self, x1, return_features=False):
        # input x1 shape: (B, 1, C, H, W)
        x1 = self.conv5(x1)
        # shape after conv5: (B, 8, C-8, H, W)
        x1 = x1.reshape(x1.shape[0], -1, self.patchsize, self.patchsize)
        x1 = self.conv6(x1)
        cls_tokens = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = x1.flatten(2)
        x1 = x1.transpose(-1, -2)
        wa = self.token_wA.expand(x1.shape[0],-1,-1)
        wa = rearrange(wa, 'b h w -> b w h')
        A = torch.einsum('bij,bjk->bik', x1, wa)
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(dim=-1)
        wv = self.token_wV.expand(x1.shape[0],-1,-1)
        VV = torch.einsum('bij,bjk->bik', x1, wv)
        T = torch.einsum('bij,bjk->bik', A, VV)
        x = torch.cat((cls_tokens, T), dim = 1) 
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        x = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], int(math.sqrt(self.FM*4)), int(math.sqrt(self.FM*4)))
        
        pooled = self.ca(x)
        pooled = pooled.reshape(pooled.shape[0], -1)
        hash_codes = self.hash_head(pooled)
        
        if return_features:
            return hash_codes, pooled
        return hash_codes
