"""SSM convolution kernels.

SSKernelDiag is the S4D kernel, a simpler algorithm for computing the kernel for the case of diagonal state matrices A.
"""

import math
import torch
import torch.nn as nn

from einops import rearrange, repeat
from opt_einsum import contract

from src.models.ssm import dplr
from src.ops.krylov import power

from src.utils.utils import get_logger

logger = get_logger(__name__)

from src.ops.vandermonde import log_vandermonde_naive

try: # Try CUDA extension
    
    from csrc.cauchy.cauchy import cauchy_mult
    has_cauchy_extension = True
    logger.info("CUDA extension for Cauchy multiplication found.")
except:
    logger.warning(
        "CUDA extension for Cauchy multiplication not found. Install by going to extensions/cauchy/ and running `python setup.py install`. This should speed up end-to-end training by 10-50%"
    )
    has_cauchy_extension = False

try:
    import pykeops
    from src.ops.vandermonde import log_vandermonde, log_vandermonde_transpose
    from pykeops.torch import Genred
    has_pykeops = True
    logger.info("Pykeops installation found.")
except ImportError:
    has_pykeops = False
    from src.ops.vandermonde import log_vandermonde_naive as log_vandermonde
    from src.ops.vandermonde import log_vandermonde_transpose_naive as log_vandermonde_transpose
    logger.warning(
        "Falling back on slow Vandermonde kernel. Install pykeops for improved memory efficiency."
    )
from src.ops.vandermonde import log_vandermonde_fast


_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()

def cauchy_naive(v, z, w, conj=True):
    """
    v: (..., N)
    z: (..., L)
    w: (..., N)
    returns: (..., L) \sum v/(z-w)
    """
    if conj:
        v = _conj(v)
        w = _conj(w)
    cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1)) # (... N L)
    return torch.sum(cauchy_matrix, dim=-2)

class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": wd}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class SSKernelDiag(OptimModule):
    """Version using (complex) diagonal state matrix (S4D)"""

    def __init__(
        self,
        A, B, C, log_dt,
        L=None,
        lr=None,
        **kwargs,
    ):

        super().__init__()

        self.L = L

        # Rank of low-rank correction
        assert A.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = A.size(-1)
        assert A.size(-2) == B.size(-3 if self.mimo else -2) # Number of independent SSMs trained
        assert self.H % A.size(-2) == 0
        self.n_ssm = A.size(-2)
        self.repeat = self.H // A.size(0)

        self.channels = C.shape[0]
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))

        # Register parameters
        if lr is None or isinstance(lr, float): lr_dict = {}
        else: lr_dict, lr = lr, None

        self.register("log_dt", log_dt, lr_dict.get('dt', lr))
        # self.register("A", _c2r(A), lr_dict.get('A', lr))
        self.register("B", _c2r(B), lr_dict.get('B', lr))
        self.register("inv_A_real", self._A_init(A.real), lr_dict.get('A', lr))
        self.register("A_imag", A.imag, lr_dict.get('A', lr))

    def _A_init(self, A_real):
        A_real = torch.clamp(A_real, max=-1e-4)
        return torch.log(-A_real)

    def _A(self):
        # Get the internal A (diagonal) parameter
        A_real = -torch.exp(self.inv_A_real)
        A = A_real + 1j * self.A_imag
        return A

    def forward(self, L, state=None, rate=1.0, u=None):
        """
        state: (B, H, N) initial state
        rate: sampling rate factor
        L: target length

        returns:
        (C, H, L) convolution kernel (generally C=1)
        (B, H, L) output from initial state
        """
        dt = torch.exp(self.log_dt) * rate # (H)
        C = _r2c(self.C) # (C H N)
        A = self._A() # (H N)

        B = _r2c(self.B)
        B = repeat(B, 't n -> 1 (v t) n', v=self.repeat)

        # Incorporate dt into A
        A = repeat(A, 't n -> (v t) n', v=self.repeat)
        dtA = A * dt.unsqueeze(-1)  # (H N)

        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)

        C = C * (1. - dtA/2).reciprocal() * dt.unsqueeze(-1) # or * dtA / A
        dA = (1. + dtA/2) / (1. - dtA/2)

        if log_vandermonde_fast is not None:
            dA_log = dA.log()
            dA_log_repeat = repeat(dA_log, 'h d -> (c h) d', c=C.shape[0])
            K = rearrange(log_vandermonde_fast(rearrange(C, 'c h d -> (c h) d'), dA_log_repeat, L),
                            '(c h) d -> c h d', c=C.shape[0])
        else:
            K = log_vandermonde_naive(C, dA.log(), L)

        K = K.view(-1, self.channels, self.H, L) # (1+B C H L)
        if state is not None:
            K_state = K[:-1, ...] # (B C H L)
        else:
            K_state = None
        
        K = K[-1, ...] # (C H L)
        return K, K_state

    def _setup_step(self):
        # These methods are organized like this to be compatible with the NPLR kernel interface
        dt = torch.exp(self.log_dt) # (H)
        B = _r2c(self.B) # (H N)
        C = _r2c(self.C) # (C H N)
        self.dC = C
        A = self._A() # (H N)

        # Incorporate dt into A
        dtA = A * dt.unsqueeze(-1)  # (H N)
        if self.disc == 'zoh':
            self.dA = torch.exp(dtA) # (H N)
            self.dB = B * (torch.exp(dtA)-1.) / A # (C H N)
        elif self.disc == 'bilinear':
            self.dA = (1. + dtA/2) / (1. - dtA/2)
            self.dB = B * (1. - dtA/2).reciprocal() * dt.unsqueeze(-1) # or * dtA / A


    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2*y.real, next_state

    def forward_state(self, u, state):
        self._setup_step()
        AL = self.dA ** u.size(-1)
        u = u.flip(-1).to(self.dA).contiguous() # (B H L)
        v = log_vandermonde_transpose(u, self.dB, self.dA.log(), u.size(-1))
        next_state = AL * state + v
        return next_state

class SSKernel(nn.Module):
    """Wrapper around SSKernel parameterizations.

    The SSKernel is expected to support the interface
    forward()
    default_state()
    _setup_step()
    step()
    """

    def __init__(
        self,
        H,
        N=64,
        L=None,
        measure="diag-lin",
        rank=1,
        channels=1,
        dt_min=0.001,
        dt_max=0.1,
        deterministic=False,
        lr=None,
        mode="diag",
        verbose=False,
        measure_args={},
        **kernel_args,
    ):
        """State Space Kernel which computes the convolution kernel $\\bar{K}$

        H: Number of independent SSM copies; controls the size of the model. Also called d_model in the config.
        N: State size (dimensionality of parameters A, B, C). Also called d_state in the config. Generally shouldn't need to be adjusted and doens't affect speed much.
        L: Maximum length of convolution kernel, if known. Should work in the majority of cases even if not known.
        measure: Options for initialization of (A, B). For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin)
        rank: Rank of low-rank correction for NPLR mode. Needs to be increased for measure "legt"
        channels: C channels turns the SSM from a 1-dim to C-dim map; can think of it having C separate "heads" per SSM. This was partly a feature to make it easier to implement bidirectionality; it is recommended to set channels=1 and adjust H to control parameters instead
        dt_min, dt_max: min and max values for the step size dt (\Delta)
        mode: 'diag' is the simpler S4D
        lr: Passing in a number (e.g. 0.001) sets attributes of SSM parameers (A, B, dt). A custom optimizer hook is needed to configure the optimizer to set the learning rates appropriately for these parameters.
        """
        super().__init__()
        self.N = N
        self.H = H
        dtype, cdtype = torch.float, torch.cfloat
        self.channels = channels
        self.mode = mode
        self.verbose = verbose
        self.kernel_args = kernel_args

        # Generate dt
        if deterministic:
            log_dt = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), H))
        else:
            log_dt = torch.rand(self.H, dtype=dtype) * (
                math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)

        # Compute the preprocessed representation
        w, P, B, V = dplr.combination(measure, self.N, rank, self.n_ssm, **measure_args)

        # Broadcast C to have H channels
        if deterministic:
            C = torch.zeros(channels, self.H, self.N, dtype=cdtype)
            C[:, :, :1] = 1.
            C = contract('hmn, chn -> chm', V.conj().transpose(-1, -2), C) # V^* C
        else:
            C = torch.randn(channels, self.H, self.N//2, dtype=cdtype)

        # Broadcast other parameters to have n_ssm copies
        assert self.n_ssm % B.size(-2) == 0 \
                and self.n_ssm % P.size(-2) == 0 \
                and self.n_ssm % w.size(-2) == 0
        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
        P = repeat(P, 'r t n -> r (v t) n', v=self.n_ssm // P.size(-2)).clone().contiguous()
        w = repeat(w, 't n -> (v t) n', v=self.n_ssm // w.size(-2)).clone().contiguous()
        C = C.contiguous()

        assert (mode == "diag")
        if not measure.startswith("diag"):
            logger.warning("Diagonal kernel (S4D) activated but initialization is not intended for S4D. Set `measure` to 'diag-lin', 'diag-inv', or 'diag-legs' for the main variants, or 'diag' for a combination of S4D-Lin and S4D-Inv.")
        C = C * repeat(B, 't n -> (v t) n', v=H//self.n_ssm)
        self.kernel = SSKernelDiag(
            w, B, C, log_dt, L=L,
            lr=lr,
            **kernel_args,
        )

    def forward(self, state=None, L=None, rate=None):
        return self.kernel(state=state, L=L, rate=rate)

    @torch.no_grad()
    def forward_state(self, u, state):
        """ Forward the state through a sequence, i.e. computes the state after passing chunk through SSM

        state: (B, H, N)
        u: (B, H, L)

        Returns: (B, H, N)
        """

        if hasattr(self.kernel, "forward_state"):
            return self.kernel.forward_state(u, state)

        dA, dB = self.kernel._setup_state() # Construct dA, dB matrices
        # dA, dB = self.kernel.dA, self.kernel.dB # (H N N) (H N)

        conj = state.size(-1) != dA.size(-1)
        if conj: state = _conj(state)

        v = contract('h n, b h l -> b h n l', dB, u.flip(-1)) # dB.unsqueeze(-1) * u.flip(-1).unsqueeze(-2)
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("h m n, b h n -> b h m", AL, state)
        next_state = next_state + v

        if conj: next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def _setup_step(self, **kwargs):
        # This method is intended to be private so that setting up an S4 module with
        # ```
        # if hasattr(module, 'setup_step'): module.setup_step()
        # ```
        # will not trigger this method multiple times
        self.kernel._setup_step(**kwargs)

    def step(self, u, state, **kwargs):
        y, state = self.kernel.step(u, state, **kwargs)
        return y, state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)

