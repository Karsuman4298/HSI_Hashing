from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
import scipy.linalg
import math

def get_hadamard_centers(num_classes, bit_length):
    n = 2**math.ceil(math.log2(max(num_classes, bit_length)))
    H = scipy.linalg.hadamard(n)
    
    if n > num_classes:
        centers = H[1:num_classes+1, :bit_length]
    else:
        centers = H[:num_classes, :bit_length]
    return torch.tensor(centers, dtype=torch.float32)

class CSQLoss(nn.Module):
    """
    CSQ-style loss that keeps class centers separated and penalizes bit collapse.

    The original implementation only optimized cosine similarity to the target center,
    which allowed the model to collapse to a near-constant sign pattern. We add a
    bit-balance regularizer so the hash bits remain near 50/50 and do not drift to a
    trivial all-positive or all-negative representation.
    """

    def __init__(
        self,
        bit_length: int,
        num_classes: int,
        scale: float = 1.0,
        bit_balance_weight: float = 0.5,
        diversity_weight: float = 0.5,
        lambda_q: float = 0.0,
    ):
        super().__init__()
        self.bit_length = int(bit_length)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.bit_balance_weight = float(bit_balance_weight)
        self.diversity_weight = float(diversity_weight)
        self.lambda_q = float(lambda_q)

        # 1. Dynamic Center Adaptive Alignment
        # We initialize with Hadamard centers, but make them learnable parameters
        # so they can dynamically shift to accommodate target domain structures.
        centers = get_hadamard_centers(num_classes, bit_length)
        self.hash_centers = nn.Parameter(centers)

    def forward(self, hash_codes: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        if hash_codes.dim() != 2:
            raise ValueError(f"Expected [B, bits] hash codes, got {tuple(hash_codes.shape)}")
        if hash_codes.size(-1) != self.bit_length:
            raise ValueError(f"Expected {self.bit_length} bits, but got {hash_codes.size(-1)}")

        target_centers = self.hash_centers.to(hash_codes.device)[targets.to(torch.long)]

        cosine = torch.sum(hash_codes * target_centers, dim=1)
        hash_norm = torch.norm(hash_codes, p=2, dim=1) + 1e-8
        center_norm = torch.norm(target_centers, p=2, dim=1) + 1e-8
        cosine_sim = cosine / (hash_norm * center_norm)

        class_loss = 1.0 - cosine_sim
        if weights is not None:
            class_loss = class_loss * weights

        # 2. Bit-Balanced Regularization to the Tanh Activation
        # L_reg = sum((|h| - 1)^2) + sum(mean(h))^2
        h = torch.tanh(hash_codes)
        
        # Quantization penalty: forces continuous features to approach +1 or -1 without information loss
        quantization_penalty = torch.sum((h.abs() - 1.0).pow(2)) / hash_codes.size(0)
        
        # Bit Balance: forces an equal distribution of ones and zeros across the hash bits
        bit_mean = h.mean(dim=0)
        bit_balance = torch.sum(bit_mean.pow(2))

        total = class_loss.mean() * self.scale + self.bit_balance_weight * bit_balance + self.lambda_q * quantization_penalty
        return total

class DPNLoss(nn.Module):
    """
    A corrected DPN-style polarization loss that pushes continuous features 
    towards discrete binary states (-1 or +1) without representation collapse,
    and includes semantic classification loss.
    """
    def __init__(self, bit_length: int, num_classes: int, scale: float = 1.0, pol_weight: float = 1.0):
        super().__init__()
        self.bit_length = int(bit_length)
        self.scale = scale
        self.pol_weight = pol_weight
        self.num_classes = int(num_classes)
        
        # Center-similarity formulation to provide semantic meaning
        centers = torch.randint(0, 2, (num_classes, bit_length), dtype=torch.float32) * 2.0 - 1.0
        self.register_buffer("hash_centers", centers)

    def forward(self, hash_codes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if hash_codes.dim() != 2:
            raise ValueError(f"Expected [B, bits], got {tuple(hash_codes.shape)}")
        if hash_codes.size(-1) != self.bit_length:
            raise ValueError(
                f"Expected {self.bit_length} bits, but got {hash_codes.size(-1)}"
            )
        
        # Classification loss (semantic similarity to centers)
        target_centers = self.hash_centers.to(hash_codes.device)[targets.to(torch.long)]
        cosine = torch.sum(hash_codes * target_centers, dim=1)
        hash_norm = torch.norm(hash_codes, p=2, dim=1) + 1e-8
        center_norm = torch.norm(target_centers, p=2, dim=1) + 1e-8
        cosine_sim = cosine / (hash_norm * center_norm)
        class_loss = 1.0 - cosine_sim
        
        # Continuous approximation of binary values
        h = torch.tanh(hash_codes)
        
        # Polarization loss: Force values to polarise toward EITHER -1 OR +1.
        pol_loss = (h.pow(2) - 1.0).pow(2).mean()
        
        loss = class_loss.mean() * self.scale + pol_loss * self.pol_weight
        return loss

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss.
    Pulls features of the same class together, pushes features of different classes apart.
    Operates on continuous logits before quantization/polarization.
    """
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-contrast cases (the diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob for numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - sim_max.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] = 1 # avoid division by zero
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Final loss
        loss = - mean_log_prob_pos.mean()
        return loss

class DSHLoss(nn.Module):
    def __init__(self, num_train: int, bit_length: int, num_classes: int, alpha: float = 0.1):
        super().__init__()
        self.m = 2 * bit_length
        self.num_classes = num_classes
        self.alpha = alpha
        self.register_buffer("U", torch.zeros(num_train, bit_length).float())
        self.register_buffer("Y", torch.zeros(num_train, num_classes).float())

    def forward(self, u: torch.Tensor, targets: torch.Tensor, ind: torch.Tensor = None, epoch: int = None) -> torch.Tensor:
        y = F.one_hot(targets.to(torch.long), num_classes=self.num_classes).float()
        
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        y_sim = (y @ self.Y.t() == 0).float()

        loss = (1 - y_sim) / 2 * dist + y_sim / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = self.alpha * (u.abs() - 1).abs().mean()

        return loss1 + loss2

class GreedyHashLoss(nn.Module):
    def __init__(self, bit_length: int, num_classes: int, alpha: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.fc = nn.Linear(bit_length, num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, u: torch.Tensor, targets: torch.Tensor, ind: torch.Tensor = None, epoch: int = None) -> torch.Tensor:
        b = GreedyHashLoss.Hash.apply(u)
        y_pre = self.fc(b)
        loss1 = self.criterion(y_pre, targets.to(torch.long))
        loss2 = self.alpha * (u.abs() - 1).pow(3).abs().mean()
        return loss1 + loss2

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

class HashNetLoss(nn.Module):
    def __init__(self, num_train: int, bit_length: int, num_classes: int, alpha: float = 0.1, step_continuation: int = 20):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.step_continuation = step_continuation
        self.register_buffer("U", torch.zeros(num_train, bit_length).float())
        self.register_buffer("Y", torch.zeros(num_train, num_classes).float())

    def forward(self, u: torch.Tensor, targets: torch.Tensor, ind: torch.Tensor = None, epoch: int = None) -> torch.Tensor:
        y = F.one_hot(targets.to(torch.long), num_classes=self.num_classes).float()
        
        scale = math.pow(1.05, epoch // self.step_continuation) if epoch is not None else 1.0
        u = torch.tanh(scale * u)

        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = self.alpha * u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        
        # avoid division by zero
        S1 = S1 if S1 > 0 else 1.0
        S0 = S0 if S0 > 0 else 1.0

        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S
        return loss

class IDHNLoss(nn.Module):
    def __init__(self, num_train: int, bit_length: int, num_classes: int, alpha: float = 0.5, gamma: float = 0.1, lambda_val: float = 0.1):
        super().__init__()
        self.q = bit_length
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_val = lambda_val
        self.register_buffer("U", torch.zeros(num_train, bit_length).float())
        self.register_buffer("Y", torch.zeros(num_train, num_classes).float())

    def forward(self, u: torch.Tensor, targets: torch.Tensor, ind: torch.Tensor = None, epoch: int = None) -> torch.Tensor:
        y = F.one_hot(targets.to(torch.long), num_classes=self.num_classes).float()
        
        u = u / (u.abs() + 1)
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = y @ self.Y.t()
        norm = y.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ self.Y.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        s = s / (norm + 0.00001)

        M = (s > 0.99).float() + (s < 0.01).float()

        inner_product = self.alpha * u @ self.U.t()

        log_loss = torch.log(1 + torch.exp(-inner_product.abs())) + inner_product.clamp(min=0) - s * inner_product
        mse_loss = (inner_product + self.q - 2 * s * self.q).pow(2)

        loss1 = (M * log_loss + self.gamma * (1 - M) * mse_loss).mean()
        loss2 = self.lambda_val * (u.abs() - 1).abs().mean()

        return loss1 + loss2

class DSPCHLoss(nn.Module):
    def __init__(self, bit_length: int, num_classes: int):
        super(DSPCHLoss, self).__init__()
        # Pre-generate unique, fixed proxy binary codes for each class
        torch.manual_seed(42)
        proxies = torch.randn(num_classes, bit_length).sign()
        self.register_buffer('class_proxies', proxies)

    def forward(self, u: torch.Tensor, targets: torch.Tensor, ind: torch.Tensor = None, epoch: int = None) -> torch.Tensor:
        # Fetch the shared target proxy for each sample based on its class
        target_proxies = self.class_proxies[targets.long()]
        # Mean Squared Error alignment to push codes toward their class proxies
        alignment_loss = F.mse_loss(u, target_proxies)
        # Quantization loss to force output values close to the discrete bounds (-1, 1)
        quantization = torch.mean((torch.abs(u) - 1) ** 2)
        return alignment_loss + 0.1 * quantization

class BatchDHNNLoss(nn.Module):
    """
    Minimizes feature distance of similar pairs, maximizes distance of dissimilar pairs.
    Operates dynamically on all pairs within a batch.
    """
    def __init__(self, bit_length: int, num_classes: int, margin=2.0):
        super(BatchDHNNLoss, self).__init__()
        self.margin = margin
        # bit_length and num_classes accepted to match signature consistency

    def forward(self, u: torch.Tensor, targets: torch.Tensor, ind: torch.Tensor = None, epoch: int = None) -> torch.Tensor:
        # Compute exact squared Euclidean distance matrix
        dist_matrix = torch.cdist(u, u, p=2).pow(2)
        
        # label_matrix[i, j] = 1 if same class, else 0
        targets = targets.contiguous().view(-1, 1)
        label_matrix = torch.eq(targets, targets.T).float()
        
        # Similar pairs loss
        loss_similar = label_matrix * dist_matrix
        
        # Dissimilar pairs loss
        loss_dissimilar = (1 - label_matrix) * torch.clamp(self.margin - torch.sqrt(dist_matrix + 1e-6), min=0.0).pow(2)
        
        # Average over all non-diagonal pairs
        mask = torch.eye(targets.size(0), device=targets.device).bool()
        loss_similar = loss_similar[~mask]
        loss_dissimilar = loss_dissimilar[~mask]
        
        return torch.mean(loss_similar + loss_dissimilar)

__all__ = ["CSQLoss", "DPNLoss", "SupConLoss", "DSHLoss", "GreedyHashLoss", "HashNetLoss", "IDHNLoss", "OrthoHashLoss", "DSPCHLoss", "BatchDHNNLoss"]

def get_imbalance_mask(sigmoid_logits, labels, nclass, threshold=0.7, imbalance_scale=-1):
    if imbalance_scale == -1:
        imbalance_scale = 1 / nclass

    mask = torch.ones_like(sigmoid_logits) * imbalance_scale

    # wan to activate the output
    mask[labels == 1] = 1

    # if predicted wrong, and not the same as labels, minimize it
    correct = (sigmoid_logits >= threshold) == (labels == 1)
    mask[~correct] = 1

    multiclass_acc = correct.float().mean()

    # the rest maintain "imbalance_scale"
    return mask, multiclass_acc

class OrthoHashLoss(nn.Module):
    def __init__(self, num_classes: int, bit_length: int, ce=1, s=8, m=0.2, m_type='cos', multiclass=False, quan=0, quan_type='cs', multiclass_loss='label_smoothing'):
        super().__init__()
        self.num_classes = num_classes
        self.bit_length = bit_length
        self.ce = ce
        self.s = s
        self.m = m
        self.m_type = m_type
        self.multiclass = multiclass
        self.quan = quan
        self.quan_type = quan_type
        self.multiclass_loss = multiclass_loss
        
        # OrthoHash paper uses orthogonal targets:
        # We sample from Bernoulli(0.5) to approximate orthogonality, then scale to {-1, +1}
        torch.manual_seed(42)
        targets = torch.bernoulli(torch.empty(num_classes, bit_length).fill_(0.5)) * 2 - 1
        self.register_buffer("hash_centers", targets.float())

    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels.long(), dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
        else:
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels.long(), dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits

        return margin_logits

    def forward(self, u: torch.Tensor, targets: torch.Tensor, ind: torch.Tensor = None, epoch: int = None) -> torch.Tensor:
        # L2 normalize continuous codes and orthogonal targets
        code_norm = F.normalize(u, p=2, dim=1)
        center_norm = F.normalize(self.hash_centers, p=2, dim=1)
        
        # Calculate scaled cosine similarity as classification logits
        logits = code_norm @ center_norm.t()
        
        labels = targets.long()
        margin_logits = self.compute_margin_logits(logits, labels)
        
        if self.multiclass:
            # We skip the implementation of multiclass here for brevity since Houston2013 is single label
            loss_ce = F.cross_entropy(margin_logits, labels)
        else:
            loss_ce = F.cross_entropy(margin_logits, labels)

        if self.quan != 0:
            if self.quan_type == 'cs':
                quantization = (1. - F.cosine_similarity(u, u.detach().sign(), dim=1))
            elif self.quan_type == 'l1':
                quantization = torch.abs(u - u.detach().sign())
            else:  # l2
                quantization = torch.pow(u - u.detach().sign(), 2)
            quantization = quantization.mean()
        else:
            quantization = torch.tensor(0.).to(u.device)

        loss = self.ce * loss_ce + self.quan * quantization
        return loss
