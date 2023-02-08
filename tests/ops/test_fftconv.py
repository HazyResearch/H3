import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange

from src.ops.fftconv import fftconv_ref, fftconv_h3_ref, fftconv_func


@pytest.mark.parametrize('output_hbl_layout', [False, True])
# @pytest.mark.parametrize('output_hbl_layout', [False])
@pytest.mark.parametrize('input_hbl_layout', [False, True])
# @pytest.mark.parametrize('input_hbl_layout', [True])
@pytest.mark.parametrize('force_fp16_output', [False, True])
# @pytest.mark.parametrize('force_fp16_output', [False])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize('dropout_prob', [None, 0.0, 0.17])
# @pytest.mark.parametrize('dropout_prob', [None])
@pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 372, 512, 784, 1024, 1134, 2048, 4096, 8192])
# @pytest.mark.parametrize('seqlen', [2048, 4096, 8192])
# There was an issue with batch_size=1 and output_hbl_layout=True where we previously had a bug
# So I'm putting batch_size=1 here in the test to make sure we don't regress.
@pytest.mark.parametrize('batch_size', [1, 4])
# @pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('gelu', [True, False])
# @pytest.mark.parametrize('gelu', [False])
def test_fftconv(gelu, batch_size, seqlen, dropout_prob, dtype, force_fp16_output,
                 input_hbl_layout, output_hbl_layout):
    if dtype == torch.bfloat16 and force_fp16_output:
        pytest.skip()
    device = 'cuda'
    rtol, atol = (1e-4, 5e-4) if dtype == torch.float32 and not force_fp16_output else (1e-3, 1e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-2
    # set seed
    torch.random.manual_seed(0)
    H = 256
    if not input_hbl_layout:
        u_fp32 = torch.randn(batch_size, H, seqlen, device=device, dtype=torch.float32, requires_grad=True)
        u = u_fp32.detach().clone().type(dtype).requires_grad_()
    else:
        u_fp32 = rearrange(torch.randn(H, batch_size, seqlen, device=device),
                      'h b l -> b h l').requires_grad_()
        u = u_fp32.detach().clone().type(dtype).requires_grad_()
    u_ref = u.detach().clone().requires_grad_()
    u_fp32_ref = u_fp32.detach().clone().requires_grad_()
    k = torch.randn(H, seqlen, device=device, requires_grad=True)
    k_rev = torch.randn(H, seqlen, device=device, requires_grad=True)
    k_ref = k.detach().clone().requires_grad_()
    k_rev_ref = k_rev.detach().clone().requires_grad_()
    D = torch.randn(H, device=device, requires_grad=True)
    D_ref = D.detach().clone().requires_grad_()
    dropout_mask = (F.dropout(torch.ones(batch_size, H, device=device), dropout_prob)
                    if dropout_prob is not None else None)

    out_ref = fftconv_ref(u_ref, k_ref, D_ref, dropout_mask, gelu=gelu, k_rev=k_rev_ref)
    if force_fp16_output:
        out_ref = out_ref.to(dtype=torch.float16)
    out_ref_fp32 = fftconv_ref(u_fp32_ref, k_ref, D_ref, dropout_mask, gelu=gelu, k_rev=k_rev_ref)
    out = fftconv_func(u, k, D, dropout_mask, gelu, force_fp16_output, output_hbl_layout, k_rev=k_rev)
    assert out.dtype == dtype if not force_fp16_output else torch.float16
    if not output_hbl_layout:
        assert out.is_contiguous()
    else:
        assert rearrange(out, 'b h l -> h b l').is_contiguous()
    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')

    g = torch.randn_like(out) / 32
    out_ref.backward(g)
    out.backward(g)

    print(f'du max diff: {(u.grad - u_ref.grad).abs().max().item()}')
    print(f'dk max diff: {(k.grad - k_ref.grad).abs().max().item()}')
    print(f'dk_rev max diff: {(k_rev.grad - k_rev_ref.grad).abs().max().item()}')
    print(f'dD max diff: {(D.grad - D_ref.grad).abs().max().item()}')

    assert u.grad.dtype == u.dtype
    assert k.grad.dtype == k.dtype
    assert k_rev.grad.dtype == k_rev.dtype
    assert D.grad.dtype == D.dtype

    if dtype == torch.float32:
        assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
    else:
        # Check that FFTConv's numerical error is at most twice the numerical error
        # of a Pytorch implementation.
        assert (out - out_ref_fp32).abs().max().item() <= 2 * (out_ref - out_ref_fp32).abs().max().item()

    assert torch.allclose(u.grad, u_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(k.grad, k_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(k_rev.grad, k_rev_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(D.grad, D_ref.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize('output_hbl_layout', [False, True])
# @pytest.mark.parametrize('output_hbl_layout', [True])
@pytest.mark.parametrize('input_hbl_layout', [False, True])
# @pytest.mark.parametrize('input_hbl_layout', [True])
# @pytest.mark.parametrize('force_fp16_output', [False, True])
@pytest.mark.parametrize('force_fp16_output', [False])
# @pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('dropout_prob', [None])
@pytest.mark.parametrize('head_dim', [1, 8])
# @pytest.mark.parametrize('head_dim', [1])
@pytest.mark.parametrize('seqlen', [1024, 2048, 4096, 8192])
# @pytest.mark.parametrize('seqlen', [2048])
@pytest.mark.parametrize('batch_size', [1, 4])
# @pytest.mark.parametrize('batch_size', [8])
@pytest.mark.parametrize('gelu', [False])
def test_fftconv_h3(gelu, batch_size, seqlen, head_dim, dropout_prob, dtype, force_fp16_output,
                    input_hbl_layout, output_hbl_layout):
    if dtype == torch.bfloat16 and force_fp16_output:
        pytest.skip()
    assert not gelu
    assert dropout_prob is None

    device = 'cuda'
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 and not force_fp16_output else (1e-2, 1e-1)
    if dtype == torch.bfloat16:
        rtol, atol = 5e-2, 5e-1
    if head_dim > 1:
        rtol *= head_dim
        atol *= head_dim
    # set seed
    torch.random.manual_seed(0)
    H = 256
    if not input_hbl_layout:
        k_fp32 = torch.randn(batch_size, H, seqlen, device=device, requires_grad=True)
        q_fp32 = torch.randn(batch_size, H, seqlen, device=device, requires_grad=True)
        v_fp32 = torch.randn(batch_size, H, seqlen, device=device, requires_grad=True)
        k = k_fp32.detach().clone().type(dtype).requires_grad_()
        q = q_fp32.detach().clone().type(dtype).requires_grad_()
        v = v_fp32.detach().clone().type(dtype).requires_grad_()
    else:
        k_fp32 = rearrange(torch.randn(H, batch_size, seqlen, device=device),
                      'h b l -> b h l').requires_grad_()
        q_fp32 = rearrange(torch.randn(H, batch_size, seqlen, device=device),
                      'h b l -> b h l').requires_grad_()
        v_fp32 = rearrange(torch.randn(H, batch_size, seqlen, device=device),
                      'h b l -> b h l').requires_grad_()
        k = k_fp32.detach().clone().type(dtype).requires_grad_()
        q = q_fp32.detach().clone().type(dtype).requires_grad_()
        v = v_fp32.detach().clone().type(dtype).requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    q_ref = q.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()
    k_fp32_ref = k_fp32.detach().clone().requires_grad_()
    q_fp32_ref = q_fp32.detach().clone().requires_grad_()
    v_fp32_ref = v_fp32.detach().clone().requires_grad_()
    ssm_kernel = torch.randn(H // head_dim, seqlen, device=device, requires_grad=True)
    ssm_kernel_rev = torch.randn(H // head_dim, seqlen, device=device, requires_grad=True)
    ssm_kernel_ref = ssm_kernel.detach().clone().requires_grad_()
    ssm_kernel_rev_ref = ssm_kernel_rev.detach().clone().requires_grad_()
    D = torch.randn(H // head_dim, device=device, requires_grad=True)
    D_ref = D.detach().clone().requires_grad_()
    dropout_mask = (F.dropout(torch.ones(batch_size, H, device=device), dropout_prob)
                    if dropout_prob is not None else None)
    out_ref = fftconv_h3_ref(k_ref, ssm_kernel_ref, D_ref, q_ref, v_ref, head_dim,
                             ssm_kernel_rev=ssm_kernel_rev_ref)
    if force_fp16_output:
        out_ref = out_ref.to(dtype=torch.float16)
    out_ref_fp32 = fftconv_h3_ref(k_fp32_ref, ssm_kernel_ref, D_ref, q_fp32_ref, v_fp32_ref, head_dim,
                             ssm_kernel_rev=ssm_kernel_rev_ref)
    out = fftconv_func(k, ssm_kernel, D, dropout_mask, gelu, force_fp16_output, output_hbl_layout,
                    v, head_dim, q, k_rev=ssm_kernel_rev)
    assert out.dtype == dtype if not force_fp16_output else torch.float16
    if not output_hbl_layout:
        assert out.is_contiguous()
    else:
        assert rearrange(out, 'b h l -> h b l').is_contiguous()
    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output max relative diff: {(out - out_ref).abs().max().item() / out_ref.abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')

    g = torch.randn_like(out) / 32
    out_ref.backward(g)
    out.backward(g)

    print(f'dk max diff: {(k.grad - k_ref.grad).abs().max().item()}')
    print(f'dq max diff: {(q.grad - q_ref.grad).abs().max().item()}')
    print(f'dv max diff: {(v.grad - v_ref.grad).abs().max().item()}')
    print(f'dssm_kernel max diff: {(ssm_kernel.grad - ssm_kernel_ref.grad).abs().max().item()}')
    print(f'dssm_kernel_rev max diff: {(ssm_kernel_rev.grad - ssm_kernel_rev_ref.grad).abs().max().item()}')
    print(f'dD max diff: {(D.grad - D_ref.grad).abs().max().item()}')

    assert k.grad.dtype == k.dtype
    assert ssm_kernel.grad.dtype == ssm_kernel.dtype
    assert D.grad.dtype == D.dtype
    # Check that FFTConv's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref_fp32).abs().max().item() <= 2 * (out_ref - out_ref_fp32).abs().max().item()
    assert torch.allclose(k.grad, k_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(q.grad, q_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(v.grad, v_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(ssm_kernel.grad, ssm_kernel_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(ssm_kernel_rev.grad, ssm_kernel_rev_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(D.grad, D_ref.grad, rtol=rtol, atol=atol)
