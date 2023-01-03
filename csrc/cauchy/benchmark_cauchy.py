import math
from functools import partial

import torch

from einops import rearrange

from cauchy import cauchy_mult_torch, cauchy_mult_keops, cauchy_mult
from benchmarks.utils import benchmark_all, benchmark_combined, benchmark_forward, benchmark_backward, pytorch_profiler


def generate_data(batch_size, N, L, symmetric=True, device='cuda'):
    if not symmetric:
        v = torch.randn(batch_size, N, dtype=torch.complex64, device=device, requires_grad=True)
        w = torch.randn(batch_size, N, dtype=torch.complex64, device=device, requires_grad=True)
        z = torch.randn(L, dtype=torch.complex64, device=device)
    else:
        assert N % 2 == 0
        v_half = torch.randn(batch_size, N // 2, dtype=torch.complex64, device=device)
        v = torch.cat([v_half, v_half.conj()], dim=-1).requires_grad_(True)
        w_half = torch.randn(batch_size, N // 2, dtype=torch.complex64, device=device)
        w = torch.cat([w_half, w_half.conj()], dim=-1).requires_grad_(True)
        z = torch.exp(1j * torch.randn(L, dtype=torch.float32, device=device))
    return v, z, w


if __name__ == '__main__':
    device = 'cuda'
    bs = 1024
    N = 64
    L = 16384

    v, z, w = generate_data(bs, N, L, symmetric=True)
    v_half = v[:, :N // 2].clone().detach().requires_grad_(True)
    w_half = w[:, :N // 2].clone().detach().requires_grad_(True)

    repeats = 30
    benchmark_all(cauchy_mult_keops, v, z, w, repeats=repeats, desc='Cauchy mult keops')
    benchmark_all(cauchy_mult, v_half, z, w_half, repeats=repeats, desc='Cauchy mult symmetric')

    pytorch_profiler(cauchy_mult, v_half, z, w_half, backward=True, verbose=True)

    # out = cauchy_mult(v_half, z, w_half)
    # g = torch.randn_like(out)
    # out.backward(g)
