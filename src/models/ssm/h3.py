import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from src.models.ssm.h3_ssm_kernel import SSKernel

try:
    from src.ops.fftconv import fftconv_func
except ImportError:
    fftconv_func = None

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class H3(nn.Module):

    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=None,
            head_dim=1,
            use_fast_fftconv=False,
            fused_bias_fc=False,
            dropout=0.0,   # Just to absorb the kwarg
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L. Set l_max=None to always use a global kernel

        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        assert d_model % head_dim == 0
        self.H = d_model // head_dim
        self.N = d_state
        self.L = l_max
        self.use_fast_fftconv = use_fast_fftconv
        if self.use_fast_fftconv:
            assert fftconv_func is not None, 'Need to install fftconv'

        if fused_bias_fc and FusedDense is None:
            raise ImportError('fused_dense is not installed')
        # linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        linear_cls = nn.Linear

        self.q_proj = linear_cls(self.d_model, self.d_model)
        self.k_proj = linear_cls(self.d_model, self.d_model)
        self.v_proj = linear_cls(self.d_model, self.d_model)

        self.ConvQ = nn.Conv1d(self.d_model, self.d_model, 3, padding=2, groups=self.d_model)
        self.ConvK = nn.Conv1d(self.d_model, self.d_model, 3, padding=2, groups=self.d_model)

        # S4D Kernel
        self.kernel = SSKernel(self.H, N=self.N, L=self.L, channels=1, **kernel_args)
        self.D = nn.Parameter(torch.randn(self.H))

        # Pointwise
        # position-wise output transform to mix features
        # Don't use FusedDense since the layout is H first
        self.output_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, u, state=None, rate=1.0, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        assert state is None
        L_og = u.size(-2)
        if self.use_fast_fftconv and L_og % 2 != 0:
            u = F.pad(u, (0, 0, 0, 1))
        L = u.size(-2)

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, round(self.L / rate))
        ssm_kernel, k_state = self.kernel(L=L_kernel, rate=rate, state=state) # (C H L) (B C H L)

        u = rearrange(u, 'b l h -> (b l) h')
        # We want q, k, v to be in fp16/bf16 if running under AMP. What's the right way to get the
        # dtype when running under AMP?
        q = self.q_proj.weight @ u.T
        q = q + self.q_proj.bias.to(q.dtype).unsqueeze(-1)
        k = self.k_proj.weight @ u.T + self.k_proj.bias.to(q.dtype).unsqueeze(-1)
        v = self.v_proj.weight @ u.T + self.v_proj.bias.to(q.dtype).unsqueeze(-1)
        q, k, v = [rearrange(x, 'h (b l) -> b h l', l=L) for x in [q, k, v]]

        
        q = self.ConvQ(q)[..., :L]
        k = self.ConvK(k)[..., :L]

        if not self.use_fast_fftconv:
            fft_size = L_kernel + L
            # kv = k * v
            kv = (rearrange(k, 'b (h d1) l -> b d1 1 h l', d1=self.head_dim)
                    * rearrange(v, 'b (h d2) l -> b 1 d2 h l', d2=self.head_dim))  # b d1 d2 h l
            kv_f = torch.fft.rfft(kv.to(dtype=ssm_kernel.dtype), n=fft_size) / fft_size
            ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)  # h L+1
            y = torch.fft.irfft(kv_f * ssm_kernel_f, n=fft_size, norm='forward')[..., :L]  # b d1 d2 h l
            y = y + kv * self.D.unsqueeze(-1)  # b d1 d2 h l
            q = rearrange(q, 'b (h d1) l -> b d1 1 h l', d1=self.head_dim)
            # einsum is way slower than multiply and then sum.
            if self.head_dim > 1:
                y = mul_sum(y, q)
                y = rearrange(y, 'b d h l -> b (d h) l')
            else:
                y = rearrange(y * q, 'b 1 1 h l -> b h l')
        else:
            dropout_mask = None
            # No GeLU after the SSM
            # Set output_hbl_layout=True since we'll be doing a matmul right after
            y = fftconv_func(k, ssm_kernel, self.D,
                             dropout_mask, False, torch.is_autocast_enabled(), True,
                             v, self.head_dim, q)

        y = rearrange(y, 'b h l -> b l h')

        y = self.output_linear(y)
        if L_og < L:
            y = y[:, :L_og, :]

        return y
