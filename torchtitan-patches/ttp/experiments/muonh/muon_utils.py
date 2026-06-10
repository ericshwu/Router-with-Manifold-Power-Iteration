"""TorchTitan Implementation of Muon"""
import math

import torch
from torch import Tensor
import torch.nn.functional as F

import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor


from typing import List, Union


def to_local(tensor: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
    if isinstance(tensor, Tensor):
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return [t.to_local() if isinstance(t, DTensor) else t for t in tensor]


def momentum_update_pre_orthogonalize(G: Tensor, M: Tensor, beta: Tensor, use_nesterov: bool = True):
    G = G.to(dtype=M.dtype)
    # Update Momentum: M_t = beta * M_{t-1} + (1 - beta) * G_t
    M.lerp_(G, 1 - beta)
    # by default, we use nesterov for all Muon variants
    # if use nesterov, use beta * M_{t} + (1 - beta) * G_t as updata
    return torch.lerp(G, M, beta) if use_nesterov else M


def msign(G: torch.Tensor):
    "ns8 with polar express coeffs"
    coeffs = [
        # # optimized for a quintic iteration.
        # # Source: https://leloykun.github.io/ponder/muon-opt-coeffs/#how-do-we-optimize-the-coefficients
        # # Numbers from: https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L44
        # (4.0848, -6.8946, 2.9270),
        # (3.9505, -6.3029, 2.6377),
        # (3.7418, -5.5913, 2.3037),
        # (2.8769, -3.1427, 1.2046),
        # (2.8366, -3.0525, 1.2012),
        # Polar Express iteration from: https://arxiv.org/abs/2505.16932
        (7.2086, -15.5131, 9.0178),
        (3.9623, -2.5813, 0.4542),
        (3.9466, -2.5765, 0.4544),
        (3.8991, -2.5671, 0.4566),
        (3.7186, -2.5308, 0.4653),
        (3.1390, -2.3073, 0.4733),
        (2.1715, -1.5246, 0.3885),
        (1.8648, -1.2224, 0.3577),
    ]
    assert G.dim() in [2, 3] and G.dtype == torch.float32

    is_2d = (G.dim() == 2)
    if G.dim() == 2:
        G = G.unsqueeze(0)
    
    X = G

    # check if transposed
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.transpose(1, 2)
        transposed = True
    
    for step in range(len(coeffs)):
        a, b, c = coeffs[step]
        # A = torch.bmm(X, X.transpose(1, 2))
        # B = b * A + c * torch.bmm(A, A)
        # X = a * X + torch.bmm(B, X)
        A = torch.bmm(X, X.transpose(1, 2))
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)

    if transposed:
        X = X.transpose(1, 2)

    # If the original input was a 2D tensor, squeeze the batch dimension out
    if is_2d:
        X = X.squeeze(0)

    return X


@torch.no_grad()
def muon_update(M: torch.Tensor, lr: torch.Tensor, scaler: str = "mup"):
    n_out, n_in = M.shape[-2:]

    M_fp32 = M.to(torch.float32)
    M_fp32 = F.normalize(M_fp32, p=2, dim=(-2, -1), eps=1e-7)

    if scaler == "moonlight":
        adjusted_ratio = 0.2 * math.sqrt(max(n_out, n_in))
        adjusted_lr = lr * adjusted_ratio
    elif scaler == "mup":
        adjusted_ratio = math.sqrt(n_out / n_in)
        adjusted_lr = lr * adjusted_ratio

    return torch.mul(msign(M_fp32), adjusted_lr)

    
@torch.no_grad()
def muonh_update(M: torch.Tensor, R: torch.Tensor, lr: torch.Tensor):
    M_fp32 = M.to(torch.float32)
    M_fp32 = F.normalize(M_fp32, p=2, dim=(-2, -1), eps=1e-7)
    U = msign(M_fp32)
    return U.mul_(lr * R / torch.linalg.norm(U, ord='fro', dim=(-2, -1), keepdim=True).clamp_min_(1e-7))


