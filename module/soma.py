from typing import Callable
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron as ner
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.base import MemoryModule
from torch.utils.checkpoint import checkpoint


def _causal_fft_convolution_impl(
    x: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    T = x.shape[0]
    kernel_length = kernel.shape[-1]
    if T <= 0 or kernel_length <= 0:
        raise ValueError("x and kernel must have non-empty time dimensions")

    linear_length = T + kernel_length - 1
    n_fft = 1 << (linear_length - 1).bit_length()
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    x_work = x.movedim(0, -1).to(dtype=work_dtype)
    kernel_work = kernel.to(device=x.device, dtype=work_dtype)

    x_f = torch.fft.rfft(x_work, n=n_fft, dim=-1)
    kernel_f = torch.fft.rfft(kernel_work, n=n_fft, dim=-1)
    # Materialize the causal crop. A plain slice would retain the complete
    # zero-padded irfft storage, which is close to 2T when P is close to T.
    y = torch.fft.irfft(x_f * kernel_f, n=n_fft, dim=-1)[..., :T].clone(
        memory_format=torch.contiguous_format
    )
    return y.movedim(-1, 0).to(dtype=x.dtype)


def _causal_fft_convolution_time_first(
    x: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Apply a causal kernel while recomputing FFT workspaces in backward."""
    if torch.is_grad_enabled() and (x.requires_grad or kernel.requires_grad):
        return checkpoint(
            _causal_fft_convolution_impl,
            x,
            kernel,
            use_reentrant=False,
        )
    return _causal_fft_convolution_impl(x, kernel)


class BaseSoma(neuron.BaseNode):

    def __init__(
        self, v_threshold: float = 1., v_reset: float = 0., 
        surrogate_function: Callable = surrogate.Sigmoid(), 
        detach_reset: bool = False, step_mode: str = "s", 
        backend: str = "torch", 
        store_v_seq: bool = False, store_v_pre_spike: bool = False
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode,
            backend, store_v_seq
        )
        self.store_v_pre_spike = store_v_pre_spike

    @property
    def store_v_pre_spike(self) -> bool:
        return self._store_v_pre_spike

    @store_v_pre_spike.setter
    def store_v_pre_spike(self, val: bool):
        self._store_v_pre_spike = val
        if val and (not hasattr(self, "v_pre_spike")):
            self.register_memory("v_pre_spike", None)

    def _single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        v_pre_spike = self.v
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike, v_pre_spike

    def single_step_forward(self, x: torch.Tensor):
        spike, v_pre_spike = self._single_step_forward(x)
        if self.store_v_pre_spike:
            self.v_pre_spike = v_pre_spike
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_pre_spike:
            v_pre_spike_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y, v_pre_spike = self._single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_pre_spike:
                v_pre_spike_seq.append(v_pre_spike)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        if self.store_v_pre_spike:
            self.v_pre_spike = torch.stack(v_pre_spike_seq)
        return torch.stack(y_seq)


class MaskedSlidingPSN(nn.Module):
    def gen_gemm_weight(self, T: int):
        weight = torch.zeros(
            [T, T],
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.order)
            length = min(end - start, self.order)
            weight[i][start: end] = self.weight[self.order - length: self.order]

        return weight


    def __init__(self, order: int, surrogate_function=surrogate.Sigmoid(), exp_init: bool=True, backend='fft'):
        super().__init__()
        if order <= 0:
            raise ValueError("order must be positive")
        if backend not in ('gemm', 'conv', 'fft'):
            raise ValueError("backend must be 'gemm', 'conv', or 'fft'")
        self.order = order
        self.backend = backend
        if self.backend in ('gemm', 'fft'):
            if exp_init:
                weight = torch.ones([order])
                for i in range(order - 2, -1, -1):
                    weight[i] = weight[i + 1] / 2.

                self.weight = nn.Parameter(weight)
            else:
                self.weight = torch.ones([1, order])
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                self.weight = nn.Parameter(self.weight[0])


            self.threshold = nn.Parameter(torch.as_tensor(-1.))
            self.surrogate_function = surrogate_function


        elif self.backend == 'conv':
            self.weight = torch.zeros([1, 1, order])
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.weight = nn.Parameter(self.weight)
            self.threshold = nn.Parameter(torch.as_tensor(1.))
            self.surrogate_function = surrogate_function


    def forward(self, x_seq: torch.Tensor):
        if self.backend == 'gemm':
            weight = self.gen_gemm_weight(x_seq.shape[0])
            h_seq = torch.addmm(self.threshold, weight, x_seq.flatten(1)).view(x_seq.shape)
            return self.surrogate_function(h_seq)

        elif self.backend == 'conv':
            # x_seq.shape = [T, N, *]
            x_seq_shape = x_seq.shape
            # [T, N, *] -> [T, N] -> [N, T] -> [N, 1, T]
            x_seq = x_seq.flatten(1).t().unsqueeze(1)
            x_seq = F.pad(x_seq, pad=(self.order - 1, 0))
            x_seq = F.conv1d(x_seq, self.weight, stride=1)
            x_seq = x_seq.squeeze(1).t().view(x_seq_shape)
            return self.surrogate_function(x_seq - self.threshold)
        elif self.backend == 'fft':
            kernel_length = min(x_seq.shape[0], self.order)
            lag_kernel = self.weight.flip(0)[:kernel_length]
            h_seq = _causal_fft_convolution_time_first(x_seq, lag_kernel)
            return self.surrogate_function(h_seq + self.threshold)
        else:
            raise NotImplementedError(self.backend)
     
    

class SSF_Quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, L, U):
        ctx.save_for_backward(input)
        ctx.L = L
        ctx.U = U
        clipped_input = torch.clamp(input, min=L, max=U)  
        return torch.floor(clipped_input)
        #return torch.trunc(clipped_input)
        #return clipped_input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        L = ctx.L
        U = ctx.U
        grad_input = grad_output.clone()
        grad_input[input < L] = 0  ##
        grad_input[input > U] = 0
        return grad_input, None, None  ##

class SSF(nn.Module):
    def __init__(self, L: int = 0, U: int = 4):
        super().__init__()
        self.U = U
        self.L = L
        #self.phi_p = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        #self.phi_s = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):

        #phi_s_clamped = torch.clamp(self.phi_s, min=1e-3)
        #x_norm = (x - self.phi_p) / phi_s_clamped
        #x_norm = x
        spike = SSF_Quant.apply(x, self.L, self.U)
        return spike

class PSNIntergerSoma_ssf(neuron.BaseNode):
    """SSF integer soma with a PSN-style causal temporal membrane kernel.

    This is the non-astro counterpart of ``AstroPSNIntergerSoma_ssf``.  It
    replaces the fixed no-reset exponential membrane recurrence with the
    sliding causal PSN weight matrix, then applies SSF directly.
    """

    def __init__(
        self,
        psn_order: int = 32,
        psn_exp_init: bool = True,
        psn_backend: str = "fft",
        psn_threshold_init: float = 0.0,
        tau: float = 2.,
        v_threshold: float = 1., v_reset: float = 0., detach_reset: bool = True,
        decay_input: bool = True, step_mode='m', backend='torch', L=0, thre=4, ssf=True,
        surrogate_function: Callable = surrogate.Sigmoid(),
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend
        )
        if psn_order <= 0:
            raise ValueError("psn_order must be positive")
        if psn_backend not in ("gemm", "conv", "fft"):
            raise ValueError("psn_backend must be 'gemm', 'conv', or 'fft'")

        self.psn_order = int(psn_order)
        self.psn_backend = psn_backend
        self.decay_input = decay_input
        self.tau = float(tau)

        if psn_exp_init:
            weight = torch.ones(self.psn_order, dtype=torch.float32)
            for i in range(self.psn_order - 2, -1, -1):
                weight[i] = weight[i + 1] / 2.
        else:
            weight = torch.empty(1, self.psn_order, dtype=torch.float32)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight = weight[0]
        self.psn_weight = nn.Parameter(weight)
        self.psn_threshold = nn.Parameter(torch.as_tensor(psn_threshold_init, dtype=torch.float32))
        self.surrogate_function = surrogate_function
        self.thre = thre
        self.qtrick = SSF(L=L, U=thre) if ssf==True else nn.GELU()

    def gen_gemm_weight(self, T: int, device=None, dtype=None):
        if device is None:
            device = self.psn_weight.device
        if dtype is None:
            dtype = self.psn_weight.dtype
        weight = torch.zeros((T, T), device=device, dtype=dtype)
        psn_weight = self.psn_weight.to(device=device, dtype=dtype)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.psn_order)
            length = min(end - start, self.psn_order)
            weight[i, start:end] = psn_weight[self.psn_order - length:self.psn_order]
        return weight

    def psn_membrane_forward(self, x: torch.Tensor):
        if self.psn_backend == "gemm":
            weight = self.gen_gemm_weight(x.shape[0], device=x.device, dtype=x.dtype)
            mem = torch.matmul(weight, x.flatten(1)).view_as(x)
            return mem + self.psn_threshold.to(device=x.device, dtype=x.dtype)

        if self.psn_backend == "fft":
            kernel_length = min(x.shape[0], self.psn_order)
            lag_kernel = self.psn_weight.flip(0)[:kernel_length]
            mem = _causal_fft_convolution_time_first(x, lag_kernel)
            return mem + self.psn_threshold.to(device=x.device, dtype=x.dtype)

        x_shape = x.shape
        x_flat = x.flatten(1).t().unsqueeze(1)
        x_flat = F.pad(x_flat, pad=(self.psn_order - 1, 0))
        kernel = self.psn_weight.to(device=x.device, dtype=x.dtype).view(1, 1, self.psn_order)
        mem = F.conv1d(x_flat, kernel, stride=1)
        mem = mem.squeeze(1).t().view(x_shape)
        return mem + self.psn_threshold.to(device=x.device, dtype=x.dtype)

    def single_step_forward(self, x: torch.Tensor):
        return self.multi_step_forward(x.unsqueeze(0))[0]

    def multi_step_forward(self, x: torch.Tensor):
        mem_seq = self.psn_membrane_forward(x)
        #mem_seq = x
        if self.thre == 1:
            output = self.surrogate_function(mem_seq - self.thre)
        else:    
            output = self.qtrick(mem_seq)
        self.v = mem_seq[-1].detach()
        self.firing_rate = output.float().mean()
        return output

class ExponentialSoma_ssf(neuron.BaseNode):
    """Parallel exponential integration followed by the existing SSF firing.

    For an input of shape ``[T, N, ...]``, the membrane is
    ``v[t] = bias + sum(gain * decay**lag * x[t-lag], lag=0..t)``.
    The three scalar parameters are shared across channels and samples;
    ``decay`` is constrained to (0, 1). The default kernel is ``2**(-lag)``,
    matching an exponentially initialized PSN soma with ``psn_order=T``.

    Each multi-step call starts from zero integration state and uses all T
    causal lags, with no spike-dependent reset or finite sliding-window cutoff.
    Only whole-sequence (``step_mode='m'``) execution is supported. ``self.v``
    records the final biased membrane for monitoring, not recurrent carryover.
    """

    def __init__(
        self,
        decay_init: float = 0.5,
        gain_init: float = 1.0,
        bias_init: float = 0.0,
        kernel_backend: str = "fft",
        thre: int = 4,
        step_mode: str = "m",
        backend: str = "torch",
        store_v_seq: bool = False,
    ):
        if not math.isfinite(decay_init) or not 0.0 < decay_init < 1.0:
            raise ValueError("decay_init must be finite and strictly between 0 and 1")
        if not math.isfinite(gain_init) or not math.isfinite(bias_init):
            raise ValueError("gain_init and bias_init must be finite")
        if kernel_backend not in ("fft", "conv", "gemm"):
            raise ValueError("kernel_backend must be 'fft', 'conv', or 'gemm'")
        if isinstance(thre, bool) or not isinstance(thre, int) or thre <= 0:
            raise ValueError("thre must be a positive integer")
        if step_mode != "m":
            raise ValueError("ExponentialSoma_ssf supports only step_mode='m'")
        super().__init__(
            step_mode=step_mode, backend=backend, store_v_seq=store_v_seq,
        )
        self.kernel_backend = kernel_backend
        self.decay_logit = nn.Parameter(torch.tensor(
            math.log(decay_init) - math.log1p(-decay_init), dtype=torch.float32,
        ))
        self.gain = nn.Parameter(torch.tensor(gain_init, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))
        self.qtrick = SSF(U=thre)

    def exponential_kernel(self, length: int, device=None, dtype=None):
        """Return lag-ordered weights, with the current-input tap first."""
        if length <= 0:
            raise ValueError("kernel length must be positive")
        if device is None:
            device = self.decay_logit.device
        if dtype is None:
            dtype = self.decay_logit.dtype
        # Generate powers in FP32 (FP64 for double inputs), including under AMP.
        work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        logit = self.decay_logit.to(device=device, dtype=work_dtype)
        eps = torch.finfo(work_dtype).eps
        decay = torch.sigmoid(logit).clamp(min=eps, max=1.0 - eps)
        lag = torch.arange(length, device=device, dtype=work_dtype)
        return self.gain.to(device=device, dtype=work_dtype) * decay.pow(lag)

    def exponential_membrane_forward(self, x: torch.Tensor):
        if x.ndim < 2 or x.shape[0] == 0:
            raise ValueError("expected a non-empty time-first input [T, N, ...]")
        if not x.is_floating_point():
            raise TypeError("exponential soma inputs must be floating-point tensors")
        T = x.shape[0]
        kernel = self.exponential_kernel(T, device=x.device, dtype=x.dtype)
        if self.kernel_backend == "fft":
            mem = _causal_fft_convolution_time_first(x, kernel)
        elif self.kernel_backend == "conv":
            x_flat = x.flatten(1).t().unsqueeze(1).to(dtype=kernel.dtype)
            mem = F.conv1d(
                F.pad(x_flat, (T - 1, 0)), kernel.flip(0).view(1, 1, T),
            )
            mem = mem.squeeze(1).t().reshape_as(x).to(dtype=x.dtype)
        else:
            t = torch.arange(T, device=x.device)
            lag = t[:, None] - t[None, :]
            weight = kernel[lag.clamp_min(0)] * (lag >= 0)
            mem = (weight @ x.flatten(1).to(dtype=kernel.dtype)).reshape_as(x)
            mem = mem.to(dtype=x.dtype)
        return mem + self.bias.to(device=x.device, dtype=x.dtype)

    def single_step_forward(self, x: torch.Tensor):
        raise NotImplementedError("ExponentialSoma_ssf requires a complete [T, N, ...] sequence")

    def multi_step_forward(self, x: torch.Tensor):
        mem_seq = self.exponential_membrane_forward(x)
        output = self.qtrick(mem_seq)
        self.v = mem_seq[-1].detach()
        if self.store_v_seq:
            self.v_seq = mem_seq
        self.firing_rate = output.float().mean()
        return output




class FullPSNIntergerSoma_ssf(neuron.BaseNode):
    """SSF integer soma with the original full PSN temporal matrix.

    This follows the PSN style in ``train_sequential_cifar.IFNode5``: a learnable
    ``Linear(T, T)`` mixes the whole time axis before the SSF quantizer.  Unlike
    ``PSNIntergerSoma_ssf``, this is not a sliding causal kernel and therefore
    requires a fixed sequence length ``T``.
    """

    def __init__(
        self,
        T: int,
        psn_bias_init: float = -1.0,
        v_threshold: float = 1., v_reset: float = 0., detach_reset: bool = True,
        step_mode='m', backend='torch', thre=4,
        surrogate_function: Callable = surrogate.Sigmoid(),
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend
        )
        if T <= 0:
            raise ValueError("T must be positive")
        self.T = int(T)
        self.fc = nn.Linear(self.T, self.T)
        nn.init.constant_(self.fc.bias, psn_bias_init)
        self.qtrick = SSF(U=thre, v_th=v_threshold)

    def psn_membrane_forward(self, x: torch.Tensor):
        if x.shape[0] != self.T:
            raise ValueError(
                f"FullPSNIntergerSoma_ssf was initialized with T={self.T}, "
                f"but got input sequence length {x.shape[0]}"
            )
        mem = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, x.flatten(1))
        return mem.view_as(x)

    def single_step_forward(self, x: torch.Tensor):
        if self.T != 1:
            raise ValueError(
                "single_step_forward is only valid when FullPSNIntergerSoma_ssf "
                "is initialized with T=1"
            )
        return self.multi_step_forward(x.unsqueeze(0))[0]

    def multi_step_forward(self, x: torch.Tensor):
        mem_seq = self.psn_membrane_forward(x)
        output = self.qtrick(mem_seq)
        self.v = mem_seq[-1].detach()
        self.firing_rate = output.float().mean()
        return output

class LIFSoma(BaseSoma):

    def __init__(
        self, tau: float = 2., decay_input: bool = True, 
        v_threshold: float = 1., v_reset: float = 0., 
        surrogate_function: Callable = surrogate.Sigmoid(), 
        detach_reset: bool = True, step_mode: str = "s", 
        backend: str = "torch", 
        store_v_seq: bool = False, store_v_pre_spike: bool = False
    ):
        if not (isinstance(tau, float) and tau >= 1.):
            return AssertionError(
                f"LIFSoma.tau should be larger than 1., "
                f"but get tau={tau}."
            )
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, 
            step_mode, backend, store_v_seq, store_v_pre_spike
        )
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.decay_input = decay_input

    def extra_repr(self):
        return (
            super().extra_repr() + 
            f", tau={self.tau}, decay_input={self.decay_input}"
        )

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge_decay_reset0(
        x: torch.Tensor, v: torch.Tensor, tau: float
    ):
        v = v + (x - v) / tau
        return v

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge_decay(
        x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float
    ):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge_no_decay_reset0(
        x: torch.Tensor, v: torch.Tensor, tau: float
    ):
        v = v  - v / tau + x
        return v

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge_no_decay(
        x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float
    ):
        v = v - (v - v_reset) / tau + x
        return v

    @staticmethod
    @torch.jit.script
    def jit_single_eval_hard_decay(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, 
        v_reset: float, tau: float
    ):
        v_pre_spike = v + (x - (v - v_reset)) / tau
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = spike * v_reset + (1. - spike) * v_pre_spike
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_single_eval_hard_no_decay(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float,
        v_reset: float, tau: float
    ):
        v_pre_spike = v - (v - v_reset) / tau + x
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = spike * v_reset + (1. - spike) * v_pre_spike
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_single_eval_soft_decay(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        v_pre_spike = v + (x - v) / tau
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = v_pre_spike - spike * v_threshold
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_single_eval_soft_no_decay(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        v_pre_spike = v - v / tau + x
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = v_pre_spike - spike * v_threshold
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_decay(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
        v_reset: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_decay_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, 
        v_reset: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_no_decay(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
        v_reset: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_no_decay_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, 
        v_reset: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_decay(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_decay_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_no_decay(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_no_decay_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    def neuronal_charge(self, x: torch.Tensor):
        tau = torch.clamp(self.tau, min=1+1e-3)
        if self.decay_input:
            if (self.v_reset is None) or (self.v_reset == 0.):
                self.v = self.jit_neuronal_charge_decay_reset0(
                    x, self.v, tau
                )
            else:
                self.v = self.jit_neuronal_charge_decay(
                    x, self.v, self.v_reset, tau
                )
        else:
            if (self.v_reset is None) or (self.v_reset == 0.):
                self.v = self.jit_neuronal_charge_no_decay_reset0(
                    x, self.v, tau
                )
            else:
                self.v = self.jit_neuronal_charge_no_decay(
                    x, self.v, self.v_reset, tau
                )

    def single_step_forward(self, x: torch.Tensor):
        tau = torch.clamp(self.tau, min=1+1e-3)
        if self.training:
            return super().single_step_forward(x)
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None: # soft reset
                if self.decay_input:
                    spike, v_pre_spike, self.v = (
                        self.jit_single_eval_soft_decay(
                            x, self.v, self.v_threshold, tau
                        )
                    )
                else:
                    spike, v_pre_spike, self.v = (
                        self.jit_single_eval_soft_no_decay(
                            x, self.v, self.v_threshold, tau
                        )
                    )
            else:
                if self.decay_input:
                    spike, v_pre_spike, self.v = (
                        self.jit_single_eval_hard_decay(
                            x, self.v, self.v_threshold, self.v_reset, tau
                        )
                    )
                else:
                    spike, v_pre_spike, self.v = (
                        self.jit_single_eval_hard_no_decay(
                            x, self.v, self.v_threshold, self.v_reset, tau
                        )
                    )
            if self.store_v_pre_spike:
                self.v_pre_spike = v_pre_spike
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        tau = torch.clamp(self.tau, min=1+1e-3)
        if self.training:
            return super().multi_step_forward(x_seq)
        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None: # soft reset
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                            self.jit_multi_eval_soft_decay_v_seq(
                                x_seq, self.v, self.v_threshold, tau
                            )
                        )
                    else:
                        spike_seq, v_pre_spike_seq, self.v = (
                            self.jit_multi_eval_soft_decay(
                                x_seq, self.v, self.v_threshold, tau
                            )
                        )
                else:
                    if self.store_v_seq:
                        spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                            self.jit_multi_eval_soft_no_decay_v_seq(
                                x_seq, self.v, self.v_threshold, tau
                            )
                        )
                    else:
                        spike_seq, v_pre_spike_seq, self.v = (
                            self.jit_multi_eval_soft_no_decay(
                                x_seq, self.v, self.v_threshold, tau
                            )
                        )
            else: # hard reset
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                            self.jit_multi_eval_hard_decay_v_seq(
                                x_seq, self.v, self.v_threshold, 
                                self.v_reset, tau
                            )
                        )
                    else:
                        spike_seq, v_pre_spike_seq, self.v = (
                            self.jit_multi_eval_hard_decay(
                                x_seq, self.v, self.v_threshold, 
                                self.v_reset, tau
                            )
                        )
                else:
                    if self.store_v_seq:
                        spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                            self.jit_multi_eval_hard_no_decay_v_seq(
                                x_seq, self.v, self.v_threshold, 
                                self.v_reset, tau
                            )
                        )
                    else:
                        spike_seq, v_pre_spike_seq, self.v = (
                            self.jit_multi_eval_hard_no_decay(
                                x_seq, self.v, self.v_threshold, 
                                self.v_reset, tau
                            )
                        )
            self.firing_rate = spike_seq.float().mean(dim=0)            
            if self.store_v_pre_spike:
                self.v_pre_spike = v_pre_spike_seq
            return spike_seq




