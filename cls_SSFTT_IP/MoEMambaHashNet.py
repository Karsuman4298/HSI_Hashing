import math
import torch
from torch import nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError(
        "MoEMambaHashNet requires the official mamba_ssm package."
    ) from exc

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class SwitchMixtureOfExperts(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        expert_output_dim,
        num_experts,
        top_k=1,
    ):
        super(SwitchMixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router: MLP to generate logits for expert selection
        self.router = nn.Linear(input_dim, num_experts)

        # Experts: a list of FeedForward networks
        self.experts = nn.ModuleList(
            [
                FeedForward(input_dim, hidden_dim, expert_output_dim)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)  # Flatten to [B*SEQLEN, dim]

        # Routing tokens to experts
        router_logits = self.router(x_flat)
        topk_logits, topk_indices = router_logits.topk(
            self.top_k, dim=1
        )
        topk_gates = F.softmax(
            topk_logits, dim=1
        )  # Normalizing the top-k logits

        # Initializing the output
        output_flat = torch.zeros(
            batch_size * seq_len,
            self.experts[0].network[-1].out_features,
            device=x.device,
        )

        # Distributing tokens to the experts and aggregating the results
        for i in range(self.top_k):
            expert_index = topk_indices[:, i]
            gate_value = topk_gates[:, i].unsqueeze(1)

            # Vectorized evaluation instead of python loop for much better speed
            expert_output = torch.zeros_like(output_flat)
            for idx in range(self.num_experts):
                mask = (expert_index == idx)
                if mask.any():
                    expert_output[mask] = self.experts[idx](x_flat[mask])

            output_flat += gate_value * expert_output

        # Reshape the output to the original input shape [B, SEQLEN, expert_output_dim]
        output = output_flat.view(batch_size, seq_len, -1)
        return output


class MoESpaMamba(nn.Module):
    def __init__(self, channels, num_experts=4, top_k=1, use_residual=True, group_num=4, use_proj=True):
        super(MoESpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(
           d_model=channels,
           d_state=16,
           d_conv=4,
           expand=2,
        )
        self.moe = SwitchMixtureOfExperts(
            input_dim=channels,
            hidden_dim=channels * 4,
            expert_output_dim=channels,
            num_experts=num_experts,
            top_k=top_k
        )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self, x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        x_flat = x_re.view(B, H * W, C)
        
        # Mamba pass
        x_mamba = self.mamba(x_flat)
        
        # MoE pass
        x_moe = self.moe(x_mamba)
        
        # Residual connection over MoE
        x_moe = x_moe + x_mamba 
        
        x_recon = x_moe.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class MoEMambaHashNet(nn.Module):
    def __init__(self, in_channels=30, hidden_dim=64, hash_bit_length=64, use_residual=True, num_experts=4, top_k=1, group_num=4):
        super(MoEMambaHashNet, self).__init__()
        
        # Initial embedding from PCA channels to hidden_dim
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )
        
        # We replace the Spatial Mamba blocks with MoE Spatial Mamba blocks
        self.mamba = nn.Sequential(
            MoESpaMamba(hidden_dim, num_experts=num_experts, top_k=top_k, use_residual=use_residual, group_num=group_num),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            MoESpaMamba(hidden_dim, num_experts=num_experts, top_k=top_k, use_residual=use_residual, group_num=group_num),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            MoESpaMamba(hidden_dim, num_experts=num_experts, top_k=top_k, use_residual=use_residual, group_num=group_num)
        )

        # Head for hashing
        self.hash_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, hash_bit_length)
        )

    def forward(self, x, mask=None, return_features=False):
        # Our input from dataset is shape (B, 1, C, H, W)
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)

        x = self.patch_embedding(x)
        x = self.mamba(x)

        hash_codes = self.hash_head(x)
        
        if return_features:
            pooled = x.mean(dim=[2,3]).flatten(1)
            return hash_codes, pooled
            
        return hash_codes
