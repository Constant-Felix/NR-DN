from typing import Callable
import torch
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

def expand_tensor_cumulative(tensor, max_value=4):

    T, B, C, H, W = tensor.shape
    # 创建一个 shape 为 [max_value, 1, 1, 1, 1, 1] 的比较向量
    steps = torch.arange(max_value, device=tensor.device).view(max_value, 1, 1, 1, 1, 1)

    # 扩展原始张量维度，便于比较 → [1, T, B, C, H, W]
    tensor_expanded = tensor.unsqueeze(0)

    # 比较：每个位置 v，生成 v 个 1，其余为 0
    binary = (steps < tensor_expanded).float()  # → shape [max_value, T, B, C, H, W]

    # 重新 reshape → [max_value * T, B, C, H, W]
    binary = binary.permute(1, 0, 2, 3, 4, 5).reshape(T * max_value, B, C, H, W)

    return binary

class IFNode5(nn.Module):
    def __init__(self, T: int, surrogate_function=surrogate.Sigmoid()):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.fc = nn.Linear(T, T)

        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        nn.init.constant_(self.fc.bias, -1)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, x_seq.flatten(1))
        spike = self.surrogate_function(h_seq)
        return spike.view(x_seq.shape)


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

class DecayPorderMaskedLinear(nn.Linear):
    def __init__(self, P: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P = P
        mask1 = torch.ones_like(self.weight.data)
        mask0 = torch.tril(mask1) * torch.triu(mask1, -(P - 1))
        self.register_buffer('mask0', mask0)
        self.register_buffer('mask1', mask1)
        self.k = 0.
        # k should be set as epoch / (epochs - 1)

    @staticmethod
    @torch.jit.script
    def gen_mask(k: float, mask0: torch.Tensor, mask1: torch.Tensor):
        return k * mask0 + (1. - k) * mask1

    @staticmethod
    @torch.jit.script
    def gen_masked_weight(weight: torch.Tensor, k: float, mask0: torch.Tensor, mask1: torch.Tensor):
        return weight * (k * mask0 + (1. - k) * mask1)

    def masked_weight(self):
        return self.gen_masked_weight(self.weight, self.k, self.mask0, self.mask1)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight * self.gen_mask(self.k, self.mask0, self.mask1), self.bias)
    
class IFNode5PorderMaskD(nn.Module):
    def __init__(self, T: int, P: int, surrogate_function=surrogate.Sigmoid()):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.fc = DecayPorderMaskedLinear(P, T, T)
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        nn.init.constant_(self.fc.bias, -1)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.masked_weight(), x_seq.flatten(1))
        spike = self.surrogate_function(h_seq)
        return spike.view(x_seq.shape)    

class AstroSomaMixin(MemoryModule):
    """Shared soma-side astrocyte state for spike-driven modulation."""

    def _init_astro_state(
        self,
        astro_lambda: float = 0.2,
        astro_trace_decay: float = 0.9,
        astro_gain: float = 0.4,
        astro_bias_gain: float = 0.0,
        astro_thre: float = 1.0,
        astro_spike_scale: float = 1.0,
        astro_pool_kernel: int = 3,
        astro_pool_mode: str = "avg",
        astro_channel_pool_when_length1: bool = False,
        astro_event_write: bool = False,
        astro_event_threshold: float = 0.4,
        astro_event_slope: float = 8.0,
        astro_delta_weight: float = 0.5,
        astro_update_interval: int = 1,
        store_c_seq: bool = False,
    ):
        astro_lambda = torch.as_tensor(astro_lambda, dtype=torch.float32)
        astro_lambda = torch.clamp(astro_lambda, 1e-3, 1.0 - 1e-3)
        astro_trace_decay = torch.as_tensor(astro_trace_decay, dtype=torch.float32)
        astro_trace_decay = torch.clamp(astro_trace_decay, 1e-3, 1.0 - 1e-3)
        self.astro_lambda_logit = nn.Parameter(torch.logit(astro_lambda))
        self.astro_trace_decay_logit = nn.Parameter(torch.logit(astro_trace_decay))
        self.astro_gain = nn.Parameter(torch.tensor(astro_gain, dtype=torch.float32))
        self.astro_bias_gain = nn.Parameter(torch.tensor(astro_bias_gain, dtype=torch.float32))
        self.astro_thre = nn.Parameter(torch.tensor(astro_thre, dtype=torch.float32))  ##
        self.astro_spike_scale = float(astro_spike_scale)
        self.astro_pool_kernel = int(astro_pool_kernel)
        self.astro_pool_mode = astro_pool_mode
        self.astro_channel_pool_when_length1 = bool(astro_channel_pool_when_length1)
        self.astro_event_write = bool(astro_event_write)
        self.astro_event_threshold = nn.Parameter(torch.tensor(astro_event_threshold, dtype=torch.float32))
        self.astro_delta_weight = nn.Parameter(torch.tensor(astro_delta_weight, dtype=torch.float32))
        self.astro_event_slope = float(astro_event_slope)
        if int(astro_update_interval) <= 0:
            raise ValueError("astro_update_interval must be positive")
        self.astro_update_interval = int(astro_update_interval)
        self.register_memory("c", 0.0)
        self.register_memory("astro_trace", 0.0)
        self.register_memory("astro_event_gate", 0.0)
        self.register_memory("firing_rate", 0.0)
        self.store_c_seq = store_c_seq

    @property
    def store_c_seq(self) -> bool:
        return self._store_c_seq

    @store_c_seq.setter
    def store_c_seq(self, val: bool):
        self._store_c_seq = val
        if val and (not hasattr(self, "c_seq")):
            self.register_memory("c_seq", None)

    def c_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.c, float):
            self.c = torch.zeros_like(x)
        if isinstance(self.astro_trace, float):
            self.astro_trace = torch.zeros_like(x)
        if isinstance(self.astro_event_gate, float):
            self.astro_event_gate = torch.zeros_like(x)    

    def astro_decode(self):
        c = torch.tanh(self.c)
        gain = 1.0 + self.astro_gain * c
        bias = self.astro_bias_gain * c
        return gain, bias

    def astro_modulate_membrane(self, membrane: torch.Tensor):
        gain, bias = self.astro_decode()
        return gain * membrane + bias

    def _astro_spike_activity(self, spike: torch.Tensor):
        scale = max(self.astro_spike_scale, 1.0)
        return torch.clamp(spike.abs() / scale, min=0.0, max=1.0)

    def _astro_pool_activity(self, activity: torch.Tensor):
        kernel = self.astro_pool_kernel
        if kernel <= 1:
            return activity       

        padding = kernel // 2
        if activity.dim() >= 4:
            if self.astro_pool_mode == "max":
                return F.max_pool2d(activity, kernel_size=kernel, stride=1, padding=padding)
            return F.avg_pool2d(
                activity, kernel_size=kernel, stride=1,
                padding=padding, count_include_pad=False
            )
        if activity.dim() == 3:
            if self.astro_channel_pool_when_length1 and activity.shape[-1] == 1:
                channel_activity = activity.transpose(1, 2)
                if self.astro_pool_mode == "max":
                    pooled = F.max_pool1d(
                        channel_activity, kernel_size=kernel, stride=1, padding=padding
                    )
                else:
                    pooled = F.avg_pool1d(
                        channel_activity, kernel_size=kernel, stride=1,
                        padding=padding, count_include_pad=False
                    )
                return pooled.transpose(1, 2)
            if self.astro_pool_mode == "max":
                return F.max_pool1d(activity, kernel_size=kernel, stride=1, padding=padding)
            return F.avg_pool1d(
                activity, kernel_size=kernel, stride=1,
                padding=padding, count_include_pad=False
            )
        return activity

    def astro_pool_spike(self, spike: torch.Tensor):
        return self._astro_pool_activity(self._astro_spike_activity(spike))

    def astro_pool_spike_sequence(self, spike_seq: torch.Tensor):
        activity = self._astro_spike_activity(spike_seq)
        if activity.dim() < 4:
            return activity

        T, N = activity.shape[:2]
        flat_activity = activity.reshape(T * N, *activity.shape[2:])
        pooled = self._astro_pool_activity(flat_activity)
        return pooled.reshape(T, N, *pooled.shape[1:])

    def astro_update_from_pooled(
        self,
        pooled_spike: torch.Tensor,
        activity_drive: torch.Tensor = None,
        block_steps: int = 1,
    ):
        if isinstance(self.astro_trace, float):
            self.astro_trace = torch.zeros_like(pooled_spike)
        if isinstance(self.c, float):
            self.c = torch.zeros_like(pooled_spike)

        block_steps = int(block_steps)
        if block_steps <= 0:
            raise ValueError("block_steps must be positive")

        trace_decay = torch.sigmoid(self.astro_trace_decay_logit)
        lam = torch.sigmoid(self.astro_lambda_logit)
        next_trace = trace_decay * self.astro_trace + pooled_spike  ##
        #next_trace = trace_decay * self.astro_trace + (1-trace_decay) * pooled_spike
        write = torch.tanh(torch.relu(next_trace - self.astro_thre.abs()))
        #block_lam = 1.0 - (1.0 - lam).pow(block_steps) ##
        block_lam = lam

        if activity_drive is None:
            activity_drive = pooled_spike.abs()

        if self.astro_event_write:
            delta_drive = (write - self.c).abs()
            event_drive = activity_drive.abs() + self.astro_delta_weight.abs() * delta_drive
            event_gate = torch.sigmoid(
                self.astro_event_slope * (event_drive - self.astro_event_threshold.abs())
            )
            #event_gate = event_gate * (event_drive > 0).to(event_gate.dtype)
        else:
            event_gate = torch.ones_like(write)

        self.astro_trace = next_trace
        self.c = (1.0 - block_lam * event_gate) * self.c + block_lam * event_gate * write
        self.astro_event_gate = event_gate.detach()
        return self.c

    def astro_update(self, spike: torch.Tensor):
        pooled_spike = self.astro_pool_spike(spike)
        self.astro_update_from_pooled(
            pooled_spike,
            activity_drive=pooled_spike.abs(),
            block_steps=1,
        )
        self.firing_rate = spike.float().mean()
        return self.c

    def astro_update_block(self, spike_seq: torch.Tensor):
        pooled_seq = self.astro_pool_spike_sequence(spike_seq)
        if pooled_seq.shape[0] == 1:
            self.astro_update_from_pooled(
                pooled_seq[0],
                activity_drive=pooled_seq[0].abs(),
                block_steps=1,
            )
            self.firing_rate = spike_seq.float().mean()
            return self.c

        trace_decay = torch.sigmoid(self.astro_trace_decay_logit)
        block_steps = pooled_seq.shape[0]
        powers = trace_decay.pow(
            torch.arange(
                block_steps - 1,
                -1,
                -1,
                device=pooled_seq.device,
                dtype=pooled_seq.dtype,
            )
        )
        powers = powers.view(block_steps, *([1] * (pooled_seq.dim() - 1)))
        #trace_drive = (powers * pooled_seq).sum(dim=0)
        trace_drive = pooled_seq.mean(dim=0)
        activity_drive = pooled_seq.mean(dim=0)
        self.astro_update_from_pooled(
            trace_drive,
            activity_drive=activity_drive,
            block_steps=block_steps,
        )
        self.firing_rate = spike_seq.float().mean()
        return self.c

    def reset_astro_state(self):
        self.c = 0.0
        self.astro_trace = 0.0
        self.astro_event_gate = 0.0
        self.firing_rate = 0.0
        if hasattr(self, "c_seq"):
            self.c_seq = None

    def detach_astro_state(self):
        if isinstance(self.c, torch.Tensor):
            self.c = self.c.detach()
        if isinstance(self.astro_trace, torch.Tensor):
            self.astro_trace = self.astro_trace.detach()


class MultiSpike8(nn.Module):

    class quant8(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.floor(torch.clamp(input, min=0, max=8))   ####改成了取整

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 8] = 0
            return grad_input

    def forward(self, x):
        return self.quant8.apply(x) 


class MultiSpike4(nn.Module):

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)
     

class IntergerSoma(neuron.BaseNode):
    def __init__(
        self, tau: float = 2.,
        v_threshold: float = 1., v_reset: float = 0.,detach_reset: bool = True,decay_input: bool = True, ##
        step_mode='m', backend='torch', thre = 4,
        surrogate_function: Callable = surrogate.Sigmoid()
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function,detach_reset,step_mode,backend
        )
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.decay_input = decay_input
        if thre == 4:
            self.qtrick = MultiSpike4()
        if thre == 8:
            self.qtrick = MultiSpike8()   
    
    def multi_step_forward(self, x: torch.Tensor):
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        tau = torch.clamp(self.tau, min=1+1e-3)
        for i in range(time_window):
            if i >= 1:
                if self.decay_input == False:
                    mem = (mem_old - spike.detach())/tau + x[i]
                else:
                    mem = (mem_old - spike.detach())/tau + x[i]*(1-1/tau)
            else:
                mem = x[i]
            spike = self.qtrick(mem)

            mem_old = mem.clone()
            output[i] = spike
        # print(output[0][0][0][0])
        #h = hook
        self.firing_rate = output.float().mean(dim=0)    
        return output


class IntergerSoma_infer(neuron.BaseNode):   ###暂时我没有改这里的tau
    def __init__(
        self, tau: float = 2.,
        v_threshold: float = 1., v_reset: float = 0.,detach_reset: bool = False,
        step_mode='m', backend='torch', 
        surrogate_function: Callable = surrogate.Sigmoid()
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function,detach_reset,step_mode,backend
        )
        self.tau = tau
        self.qtrick = MultiSpike4()
    
    def multi_step_forward(self, x: torch.Tensor):
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) / self.tau + x[i]

            else:
                mem = x[i]
            spike = self.qtrick(mem)

            mem_old = mem.clone()
            output[i] = spike

        spike = expand_tensor_cumulative(output)

        return spike
    
import torch
import torch.nn as nn

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

class IntergerSoma_ssf(neuron.BaseNode): # 假设基于 SpikingJelly 或类似框架的基类
    def __init__(
        self, tau: float = 2.,
        v_threshold: float = 1., v_reset: float = 0., detach_reset: bool = False, decay_input: bool = True,
        step_mode='m', backend='torch', thre = 4,
        surrogate_function: Callable = surrogate.Sigmoid()
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend
        )
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.decay_input = decay_input
        
        # 替换原有的 MultiSpike 为 SSF 机制
        self.qtrick = SSF(U=thre)

    def parallel_membrane_forward(self, x: torch.Tensor, tau: torch.Tensor):
        """Compute the no-reset membrane recurrence for all timesteps at once.

        This is equivalent to the previous serial recurrence in this class:
        mem[0] = x[0]
        mem[t] = beta * mem[t - 1] + drive[t], beta = 1 / tau.
        Because SSF does not feed the spike/reset result back into mem[t + 1],
        the recurrence is linear and can be written as a causal matrix product.
        """
        T = x.shape[0]
        beta = (1.0 / tau).to(device=x.device, dtype=x.dtype)
        if self.decay_input:
            drive = x * (1.0 - beta)
            drive = drive.clone()
            drive[0] = x[0]
        else:
            drive = x

        t = torch.arange(T, device=x.device, dtype=x.dtype)
        diff = t[:, None] - t[None, :]
        mask = diff >= 0
        weights = torch.zeros((T, T), device=x.device, dtype=x.dtype)
        weights = weights.masked_scatter(mask, beta.pow(diff[mask]).to(x.dtype))
        mem = torch.matmul(weights, drive.flatten(1)).view_as(x)
        return mem
    
    def multi_step_forward(self, x: torch.Tensor):
        tau = torch.clamp(self.tau, min=1+1e-3)
        mem_seq = self.parallel_membrane_forward(x, tau)
        # 这里输入给 SSF 的是原始膜电位，SSF 内部会自动进行平移和缩放
        output = self.qtrick(mem_seq)
        self.v = mem_seq[-1].detach()
        self.firing_rate = output.float().mean()
        return output


class AstroIntergerSoma(IntergerSoma, AstroSomaMixin):
    """Integer soma with spike-driven astrocyte modulation on membrane."""

    def __init__(
        self, tau: float = 2.,
        v_threshold: float = 1., v_reset: float = 0., detach_reset: bool = False,
        decay_input: bool = True, step_mode='m', backend='torch', thre=4,
        surrogate_function: Callable = surrogate.Sigmoid(),
        astro_lambda: float = 0.1, astro_trace_decay: float = 0.9,
        astro_gain: float = 1.0, astro_bias_gain: float = 0.0,
        astro_pool_kernel: int = 3, astro_pool_mode: str = "avg",
        astro_event_write: bool = False, astro_event_threshold: float = 0.05,
        astro_event_slope: float = 10.0, astro_delta_weight: float = 0.5,
        store_c_seq: bool = False,
    ):
        super().__init__(
            tau=tau, v_threshold=v_threshold, v_reset=v_reset,
            detach_reset=detach_reset, decay_input=decay_input,
            step_mode=step_mode, backend=backend, thre=thre,
            surrogate_function=surrogate_function,
        )
        self._init_astro_state(
            astro_lambda=astro_lambda,
            astro_trace_decay=astro_trace_decay,
            astro_gain=astro_gain,
            astro_bias_gain=astro_bias_gain,
            astro_spike_scale=float(thre),
            astro_pool_kernel=astro_pool_kernel,
            astro_pool_mode=astro_pool_mode,
            astro_event_write=astro_event_write,
            astro_event_threshold=astro_event_threshold,
            astro_event_slope=astro_event_slope,
            astro_delta_weight=astro_delta_weight,
            store_c_seq=store_c_seq,
        )

    def multi_step_forward(self, x: torch.Tensor):
        self.c_float_to_tensor(x[0])
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        c_seq = [] if self.store_c_seq else None
        tau = torch.clamp(self.tau, min=1 + 1e-3)

        for i in range(x.shape[0]):
            if i >= 1:
                if self.decay_input == False:
                    mem = (mem_old - spike.detach()) / tau + x[i]
                else:
                    mem = (mem_old - spike.detach()) / tau + x[i] * (1 - 1 / tau)
            else:
                mem = x[i]
            mem = self.astro_modulate_membrane(mem)
            spike = self.qtrick(mem)
            self.astro_update(spike)
            if self.store_c_seq:
                c_seq.append(self.c)

            mem_old = mem.clone()
            output[i] = spike

        self.v = mem_old.detach()
        self.firing_rate = output.float().mean()
        if self.store_c_seq:
            self.c_seq = torch.stack(c_seq)
        return output


class AstroIntergerSoma_ssf(IntergerSoma_ssf, AstroSomaMixin):
    """SSF integer soma with spike-driven astrocyte modulation on membrane."""

    def __init__(
        self, tau: float = 2.,
        v_threshold: float = 1., v_reset: float = 0., detach_reset: bool = True,
        decay_input: bool = True, step_mode='m', backend='torch', thre=4,
        surrogate_function: Callable = surrogate.Sigmoid(),
        astro_lambda: float = 0.2, astro_trace_decay: float = 0.9,
        astro_gain: float = 0.5, astro_bias_gain: float = 0.0, astro_thre: float = 0.4,
        astro_pool_kernel: int = 3, astro_pool_mode: str = "avg",
        astro_event_write: bool = False, astro_event_threshold: float = 0.5,
        astro_event_slope: float = 8.0, astro_delta_weight: float = 0.5,
        store_c_seq: bool = False,
    ):
        super().__init__(
            tau=tau, v_threshold=v_threshold, v_reset=v_reset,
            detach_reset=detach_reset, decay_input=decay_input,
            step_mode=step_mode, backend=backend, thre=thre,
            surrogate_function=surrogate_function,
        )
        self._init_astro_state(
            astro_lambda=astro_lambda,
            astro_trace_decay=astro_trace_decay,
            astro_gain=astro_gain,
            astro_bias_gain=astro_bias_gain,
            astro_thre=astro_thre,
            astro_spike_scale=float(thre),
            astro_pool_kernel=astro_pool_kernel,
            astro_pool_mode=astro_pool_mode,
            astro_event_write=astro_event_write,
            astro_event_threshold=astro_event_threshold,
            astro_event_slope=astro_event_slope,
            astro_delta_weight=astro_delta_weight,
            store_c_seq=store_c_seq,
        )

    def multi_step_forward(self, x: torch.Tensor):
        self.c_float_to_tensor(x[0])
        tau = torch.clamp(self.tau, min=1 + 1e-3)
        mem_seq = self.parallel_membrane_forward(x, tau)
        output = torch.zeros_like(x)
        modulated_mem = []
        c_seq = [] if self.store_c_seq else None

        for i in range(x.shape[0]):
            mem = self.astro_modulate_membrane(mem_seq[i])
            spike = self.qtrick(mem)
            self.astro_update(spike)
            if self.store_c_seq:
                c_seq.append(self.c)

            modulated_mem.append(mem)
            output[i] = spike

        self.v = modulated_mem[-1].detach()
        self.firing_rate = output.float().mean()
        if self.store_c_seq:
            self.c_seq = torch.stack(c_seq)
        return output
    
import math

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

class AstroMaskedSlidingPSN(AstroSomaMixin):
    """MaskedSlidingPSN with spike-driven astrocyte membrane modulation."""

    def __init__(
        self,
        order: int,
        surrogate_function=surrogate.Sigmoid(),
        exp_init: bool = False,
        backend: str = 'gemm',
        astro_lambda: float = 0.2,
        astro_trace_decay: float = 0.9,
        astro_gain: float = 0.5,
        astro_bias_gain: float = 0.0,
        astro_thre: float = 0.4,
        astro_spike_scale: float = 1.0,
        astro_pool_kernel: int = 3,
        astro_pool_mode: str = "avg",
        astro_event_write: bool = False,
        astro_event_threshold: float = 0.05,
        astro_event_slope: float = 10.0,
        astro_delta_weight: float = 0.5,
        astro_update_interval: int = 1,
        store_c_seq: bool = False,
    ):
        super().__init__()
        if order <= 0:
            raise ValueError("order must be positive")
        if backend not in ("gemm", "conv"):
            raise ValueError("backend must be 'gemm' or 'conv'")
        self.order = int(order)
        self.psn_backend = backend
        self.surrogate_function = surrogate_function

        if self.psn_backend == 'gemm':
            if exp_init:
                weight = torch.ones([self.order])
                for i in range(self.order - 2, -1, -1):
                    weight[i] = weight[i + 1] / 2.
                self.weight = nn.Parameter(weight)
            else:
                weight = torch.ones([1, self.order])
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                self.weight = nn.Parameter(weight[0])
            self.threshold = nn.Parameter(torch.as_tensor(-1.)) ##
        else:
            weight = torch.zeros([1, 1, self.order])
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self.weight = nn.Parameter(weight)
            self.threshold = nn.Parameter(torch.as_tensor(1.)) ##

        self._init_astro_state(
            astro_lambda=astro_lambda,
            astro_trace_decay=astro_trace_decay,
            astro_gain=astro_gain,
            astro_bias_gain=astro_bias_gain,
            astro_thre=astro_thre,
            astro_spike_scale=astro_spike_scale,
            astro_pool_kernel=astro_pool_kernel,
            astro_pool_mode=astro_pool_mode,
            astro_event_write=astro_event_write,
            astro_event_threshold=astro_event_threshold,
            astro_event_slope=astro_event_slope,
            astro_delta_weight=astro_delta_weight,
            astro_update_interval=astro_update_interval,
            store_c_seq=store_c_seq,
        )

    def gen_gemm_weight(self, T: int):
        weight = torch.zeros([T, T], device=self.weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.order)
            length = min(end - start, self.order)
            weight[i][start:end] = self.weight[self.order - length:self.order]
        return weight

    def psn_membrane_forward(self, x_seq: torch.Tensor):
        if self.psn_backend == 'gemm':
            weight = self.gen_gemm_weight(x_seq.shape[0]).to(
                device=x_seq.device, dtype=x_seq.dtype
            )
            threshold = self.threshold.to(device=x_seq.device, dtype=x_seq.dtype)
            h_seq = torch.addmm(threshold, weight, x_seq.flatten(1))
            return h_seq.view(x_seq.shape)

        if self.psn_backend == 'conv':
            x_shape = x_seq.shape
            x_flat = x_seq.flatten(1).t().unsqueeze(1)
            x_flat = F.pad(x_flat, pad=(self.order - 1, 0))
            kernel = self.weight.to(device=x_seq.device, dtype=x_seq.dtype)
            mem = F.conv1d(x_flat, kernel, stride=1)
            mem = mem.squeeze(1).t().view(x_shape)
            threshold = self.threshold.to(device=x_seq.device, dtype=x_seq.dtype)
            return mem - threshold

        raise NotImplementedError(self.psn_backend)

    def single_step_forward(self, x: torch.Tensor):
        return self.forward(x.unsqueeze(0))[0]

    def forward(self, x_seq: torch.Tensor):
        self.c_float_to_tensor(x_seq[0])
        mem_seq = self.psn_membrane_forward(x_seq)
        output = torch.zeros_like(x_seq)
        c_seq = [] if self.store_c_seq else None
        last_mem = None

        S = self.astro_update_interval
        if S <= 1:
            for i in range(x_seq.shape[0]):
                mem = self.astro_modulate_membrane(mem_seq[i]-self.threshold)
                spike = self.surrogate_function(mem+self.threshold)
                self.astro_update(spike)
                if self.store_c_seq:
                    c_seq.append(self.c)

                last_mem = mem
                output[i] = spike
        else:
            for start in range(0, x_seq.shape[0], S):
                end = min(start + S, x_seq.shape[0])
                mem = self.astro_modulate_membrane(mem_seq[start:end]-self.threshold)
                spike = self.surrogate_function(mem+self.threshold)
                self.astro_update_block(spike)
                if self.store_c_seq:
                    c_seq.extend([self.c] * (end - start))

                last_mem = mem[-1]
                output[start:end] = spike

        self.v = last_mem.detach()
        self.firing_rate = output.float().mean()
        if self.store_c_seq:
            self.c_seq = torch.stack(c_seq)
        return output

class AstroPSNIntergerSoma_ssf(neuron.BaseNode, AstroSomaMixin):
    """Astro-SSF soma whose pre-SSF membrane is generated by a PSN time kernel.

    This class keeps the soma-side astrocyte loop from ``AstroIntergerSoma_ssf``
    but replaces the fixed exponential no-reset recurrence with the causal
    sliding temporal matrix used by ``MaskedSlidingPSN``.
    """

    def __init__(
        self,
        psn_order: int = 32,
        psn_exp_init: bool = False,
        psn_backend: str = "gemm",
        psn_threshold_init: float = 0.0,
        tau: float = 2.,
        v_threshold: float = 1., v_reset: float = 0., detach_reset: bool = True,
        decay_input: bool = True, step_mode='m', backend='torch', thre=4,
        surrogate_function: Callable = surrogate.Sigmoid(),
        astro_lambda: float = 0.2, astro_trace_decay: float = 0.9,
        astro_gain: float = 0.5, astro_bias_gain: float = 0.0, astro_thre: float = 0.4,
        astro_pool_kernel: int = 3, astro_pool_mode: str = "avg",
        astro_channel_pool_when_length1: bool = False,
        astro_event_write: bool = False, astro_event_threshold: float = 0.05,
        astro_event_slope: float = 10.0, astro_delta_weight: float = 0.5,
        astro_update_interval: int = 1,  ##
        store_c_seq: bool = False,
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend
        )
        if psn_order <= 0:
            raise ValueError("psn_order must be positive")
        if psn_backend not in ("gemm", "conv"):
            raise ValueError("psn_backend must be 'gemm' or 'conv'")

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
        self.qtrick = SSF(U=thre, v_th=v_threshold)

        self._init_astro_state(
            astro_lambda=astro_lambda,
            astro_trace_decay=astro_trace_decay,
            astro_gain=astro_gain,
            astro_bias_gain=astro_bias_gain,
            astro_thre=astro_thre,
            astro_spike_scale=float(thre),
            astro_pool_kernel=astro_pool_kernel,
            astro_pool_mode=astro_pool_mode,
            astro_channel_pool_when_length1=astro_channel_pool_when_length1,
            astro_event_write=astro_event_write,
            astro_event_threshold=astro_event_threshold,
            astro_event_slope=astro_event_slope,
            astro_delta_weight=astro_delta_weight,
            astro_update_interval=astro_update_interval,
            store_c_seq=store_c_seq,
        )

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
        self.c_float_to_tensor(x[0])
        mem_seq = self.psn_membrane_forward(x)
        output = torch.zeros_like(x)
        c_seq = [] if self.store_c_seq else None
        last_mem = None

        S = self.astro_update_interval
        if S <= 1:
            for i in range(x.shape[0]):
                mem = self.astro_modulate_membrane(mem_seq[i])
                spike = self.qtrick(mem)
                self.astro_update(spike)
                if self.store_c_seq:
                    c_seq.append(self.c)

                last_mem = mem
                output[i] = spike
        else:
            for start in range(0, x.shape[0], S):
                end = min(start + S, x.shape[0])
                mem = self.astro_modulate_membrane(mem_seq[start:end])
                spike = self.qtrick(mem)
                self.astro_update_block(spike)
                if self.store_c_seq:
                    c_seq.extend([self.c] * (end - start))

                last_mem = mem[-1]
                output[start:end] = spike

        self.v = last_mem.detach()
        self.firing_rate = output.float().mean()
        if self.store_c_seq:
            self.c_seq = torch.stack(c_seq)
        return output

class SelectiveAstroPSNIntergerSoma_ssf(AstroPSNIntergerSoma_ssf):
    """Astro-PSN-SSF soma with selective membrane commits.

    The PSN and astrocyte path first produces a membrane proposal. A per-element
    binary gate then decides whether that proposal is committed to the soma
    membrane state. Astrocyte updates can optionally see only committed spikes,
    so carried timesteps do not repeatedly write to the slow astrocyte state.
    """

    def __init__(
        self,
        psn_order: int = 32,
        psn_exp_init: bool = False,
        psn_backend: str = "gemm",
        psn_threshold_init: float = 0.0,
        tau: float = 2.,
        v_threshold: float = 1., v_reset: float = 0., detach_reset: bool = True,
        decay_input: bool = True, step_mode='m', backend='torch', thre=4,
        surrogate_function: Callable = surrogate.Sigmoid(),
        astro_lambda: float = 0.2, astro_trace_decay: float = 0.9,
        astro_gain: float = 0.5, astro_bias_gain: float = 0.0, astro_thre: float = 0.4,
        astro_pool_kernel: int = 3, astro_pool_mode: str = "avg",
        astro_channel_pool_when_length1: bool = False,
        astro_event_write: bool = False, astro_event_threshold: float = 0.05,
        astro_event_slope: float = 10.0, astro_delta_weight: float = 0.5,
        astro_update_interval: int = 1,
        store_c_seq: bool = False,
        su_gate_threshold: float = 0.05,
        su_gate_slope: float = 8.0,
        su_gate_learnable: bool = True,
        su_force_first_update: bool = True,
        su_gated_astro_spike: bool = True,
        store_su_gate_seq: bool = False,
    ):
        super().__init__(
            psn_order=psn_order,
            psn_exp_init=psn_exp_init,
            psn_backend=psn_backend,
            psn_threshold_init=psn_threshold_init,
            tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            detach_reset=detach_reset,
            decay_input=decay_input,
            step_mode=step_mode,
            backend=backend,
            thre=thre,
            surrogate_function=surrogate_function,
            astro_lambda=astro_lambda,
            astro_trace_decay=astro_trace_decay,
            astro_gain=astro_gain,
            astro_bias_gain=astro_bias_gain,
            astro_thre=astro_thre,
            astro_pool_kernel=astro_pool_kernel,
            astro_pool_mode=astro_pool_mode,
            astro_channel_pool_when_length1=astro_channel_pool_when_length1,
            astro_event_write=astro_event_write,
            astro_event_threshold=astro_event_threshold,
            astro_event_slope=astro_event_slope,
            astro_delta_weight=astro_delta_weight,
            astro_update_interval=astro_update_interval,
            store_c_seq=store_c_seq,
        )
        if su_gate_threshold < 0:
            raise ValueError("su_gate_threshold must be non-negative")
        if su_gate_slope <= 0:
            raise ValueError("su_gate_slope must be positive")

        threshold = torch.tensor(float(su_gate_threshold), dtype=torch.float32)
        if su_gate_learnable:
            self.su_gate_threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("su_gate_threshold", threshold)
        self.su_gate_slope = float(su_gate_slope)
        self.su_force_first_update = bool(su_force_first_update)
        self.su_gated_astro_spike = bool(su_gated_astro_spike)
        self.store_su_gate_seq = bool(store_su_gate_seq)
        self.register_memory("su_update_rate", 0.0)
        self.register_memory("su_gate_seq", None)

    def _init_committed_membrane(self, ref: torch.Tensor):
        if isinstance(self.v, torch.Tensor) and self.v.shape == ref.shape:
            return self.v.to(device=ref.device, dtype=ref.dtype), False

        if isinstance(self.v, torch.Tensor):
            init_value = self.v_reset if self.v_reset is not None else 0.0
        else:
            init_value = self.v
        return torch.full_like(ref, float(init_value)), True

    def _selective_gate(
        self,
        proposal: torch.Tensor,
        prev_mem: torch.Tensor,
        force_update: bool = False,
    ):
        if force_update:
            return torch.ones_like(proposal)

        threshold = self.su_gate_threshold.abs().to(
            device=proposal.device, dtype=proposal.dtype
        )
        delta = (proposal - prev_mem).abs()
        soft_gate = torch.sigmoid(self.su_gate_slope * (delta - threshold))
        hard_gate = (delta > threshold).to(dtype=proposal.dtype)
        return hard_gate - soft_gate.detach() + soft_gate

    def multi_step_forward(self, x: torch.Tensor):
        self.c_float_to_tensor(x[0])
        mem_seq = self.psn_membrane_forward(x)
        output = torch.zeros_like(x)
        c_seq = [] if self.store_c_seq else None
        gate_seq = [] if self.store_su_gate_seq else None

        prev_mem, force_first_update = self._init_committed_membrane(mem_seq[0])
        last_mem = prev_mem

        S = self.astro_update_interval
        for start in range(0, x.shape[0], S):
            end = min(start + S, x.shape[0])
            proposal_block = self.astro_modulate_membrane(mem_seq[start:end])
            astro_spike_block = []

            for offset, proposal in enumerate(proposal_block):
                force_update = (
                    self.su_force_first_update
                    and force_first_update
                    and start == 0
                    and offset == 0
                )
                gate = self._selective_gate(
                    proposal, prev_mem, force_update=force_update
                )
                mem = prev_mem + gate * (proposal - prev_mem)
                spike = self.qtrick(mem)
                astro_spike = gate * spike if self.su_gated_astro_spike else spike

                output[start + offset] = spike
                astro_spike_block.append(astro_spike)
                if self.store_su_gate_seq:
                    gate_seq.append(gate.detach())

                prev_mem = mem
                last_mem = mem

            astro_spike_block = torch.stack(astro_spike_block)
            self.astro_update_block(astro_spike_block)
            if self.store_c_seq:
                c_seq.extend([self.c] * (end - start))

        self.v = last_mem.detach()
        self.firing_rate = output.float().mean()
        if self.store_c_seq:
            self.c_seq = torch.stack(c_seq)
        if self.store_su_gate_seq:
            self.su_gate_seq = torch.stack(gate_seq)
            self.su_update_rate = self.su_gate_seq.float().mean().detach()
        else:
            self.su_gate_seq = None
            self.su_update_rate = 0.0
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

class AstroFullPSNIntergerSoma_ssf(FullPSNIntergerSoma_ssf, AstroSomaMixin):
    """Astro-SSF soma with the original full PSN temporal matrix.

    This is the soma-astro counterpart of ``FullPSNIntergerSoma_ssf``.  The
    full ``Linear(T, T)`` PSN first generates the membrane sequence, then the
    astrocyte state modulates each timestep before SSF and is updated by the
    produced SSF spike.
    """

    def __init__(
        self,
        T: int,
        psn_bias_init: float = 0.0,
        v_threshold: float = 1., v_reset: float = 0., detach_reset: bool = True,
        step_mode='m', backend='torch', thre=4,
        surrogate_function: Callable = surrogate.Sigmoid(),
        astro_lambda: float = 0.2, astro_trace_decay: float = 0.9,
        astro_gain: float = 0.5, astro_bias_gain: float = 0.0, astro_thre: float = 0.4,
        astro_pool_kernel: int = 3, astro_pool_mode: str = "avg",
        astro_event_write: bool = False, astro_event_threshold: float = 0.05,
        astro_event_slope: float = 10.0, astro_delta_weight: float = 0.5,
        store_c_seq: bool = False,
    ):
        super().__init__(
            T=T,
            psn_bias_init=psn_bias_init,
            v_threshold=v_threshold,
            v_reset=v_reset,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            thre=thre,
            surrogate_function=surrogate_function,
        )
        self._init_astro_state(
            astro_lambda=astro_lambda,
            astro_trace_decay=astro_trace_decay,
            astro_gain=astro_gain,
            astro_bias_gain=astro_bias_gain,
            astro_thre=astro_thre,
            astro_spike_scale=float(thre),
            astro_pool_kernel=astro_pool_kernel,
            astro_pool_mode=astro_pool_mode,
            astro_event_write=astro_event_write,
            astro_event_threshold=astro_event_threshold,
            astro_event_slope=astro_event_slope,
            astro_delta_weight=astro_delta_weight,
            store_c_seq=store_c_seq,
        )

    def multi_step_forward(self, x: torch.Tensor):
        self.c_float_to_tensor(x[0])
        mem_seq = self.psn_membrane_forward(x)
        output = torch.zeros_like(x)
        c_seq = [] if self.store_c_seq else None
        last_mem = None

        for i in range(x.shape[0]):
            mem = self.astro_modulate_membrane(mem_seq[i])
            spike = self.qtrick(mem)
            self.astro_update(spike)
            if self.store_c_seq:
                c_seq.append(self.c)

            last_mem = mem
            output[i] = spike

        self.v = last_mem.detach()
        self.firing_rate = output.float().mean()
        if self.store_c_seq:
            self.c_seq = torch.stack(c_seq)
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


class Triangle(torch.autograd.Function):
    """Triangle surrogate copied from ASRC-SNN."""

    @staticmethod
    def forward(ctx, input, gamma=1.0):
        out = input.ge(0.0).float()
        gamma_tensor = input.new_tensor([gamma])
        ctx.save_for_backward(input, out, gamma_tensor)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, _, gamma_tensor = ctx.saved_tensors
        gamma = gamma_tensor[0].item()
        triangle = (gamma - input.abs()).clamp(min=0.0) / (gamma * gamma)
        return grad_output * triangle, None


class LIFNODE(MemoryModule):
    """LIFNODE from ASRC-SNN, using this project's MemoryModule."""

    def __init__(
        self,
        decay_para=0.5,
        v_threshold=1.0,
        surrogate_function=Triangle.apply,
        hard_reset=False,
        detach_reset=False,
        step_mode="s",
        channel=None,
    ):
        super().__init__()
        del channel
        if not 0.0 <= float(decay_para) < 1.0:
            raise ValueError("decay_para must be in [0, 1)")
        if step_mode not in {"s", "m"}:
            raise ValueError("step_mode must be 's' or 'm'")

        self.register_memory("v", 0.0)
        self.register_buffer(
            "decay_para", torch.tensor(float(decay_para), dtype=torch.float32)
        )
        self.v_threshold = float(v_threshold)
        self.surrogate_function = surrogate_function
        self.hard_reset = bool(hard_reset)
        self.detach_reset = bool(detach_reset)
        self.step_mode = step_mode

    def single_step_forward(self, x: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.full_like(x, self.v)

        self.v = self.v * self.decay_para + x
        spike = self.surrogate_function(self.v - self.v_threshold)
        spike_for_reset = spike.detach() if self.detach_reset else spike
        if self.hard_reset:
            self.v = self.v * (1.0 - spike_for_reset)
        else:
            self.v = self.v - spike_for_reset * self.v_threshold
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if x_seq.dim() != 3:
            raise ValueError("LIFNODE expects [T, N, C] in multi-step mode")
        return torch.stack(
            [self.single_step_forward(x_seq[t]) for t in range(x_seq.shape[0])]
        )

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self.single_step_forward(x)
        return self.multi_step_forward(x)


class skip_RecurrentContainer(MemoryModule):
    """SRC container; multi-step input is adapted to project shape [T, N, C]."""

    def __init__(self, sub_module: nn.Module, step_mode="s", hid_dim=64, skip=21):
        super().__init__()
        if getattr(sub_module, "step_mode", "s") != "s":
            raise ValueError("sub_module must use single-step mode")
        if int(skip) <= 0:
            raise ValueError("skip must be positive")

        self.step_mode = step_mode
        self.sub_module = sub_module
        self.skip = int(skip)
        self.mlp = nn.Linear(int(hid_dim), int(hid_dim), bias=False)
        self.register_memory("y", None)
        nn.init.orthogonal_(self.mlp.weight, gain=1.0)

    def single_step_forward(self, x: torch.Tensor):
        if self.y is None:
            self.y = x.new_zeros(self.skip, x.shape[0], x.shape[1])

        spike = self.sub_module(self.mlp(self.y[-1]) + x)
        self.y = torch.roll(self.y, 1, 0)
        self.y[0] = spike
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if x_seq.dim() != 3:
            raise ValueError("SRC expects [T, N, C]")
        return torch.stack(
            [self.single_step_forward(x_seq[t]) for t in range(x_seq.shape[0])]
        )

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self.single_step_forward(x)
        return self.multi_step_forward(x)


class adskip_RecurrentContainer(MemoryModule):
    """ASRC container; multi-step input is adapted to project shape [T, N, C]."""

    def __init__(
        self,
        sub_module: nn.Module,
        step_mode="s",
        hid_dim=64,
        skip=21,
        skip_para=None,
        de_tau=0.96,
    ):
        super().__init__()
        if getattr(sub_module, "step_mode", "s") != "s":
            raise ValueError("sub_module must use single-step mode")
        if int(skip) <= 0:
            raise ValueError("skip must be positive")

        self.step_mode = step_mode
        self.sub_module = sub_module
        self.skip = int(skip)
        self.tau = nn.Parameter(torch.ones(1), requires_grad=False)
        self.de_tau = float(de_tau)
        if skip_para is None:
            self.skip_para = nn.Parameter(torch.zeros(self.skip, 1, 1))
        else:
            self.skip_para = skip_para
        self.mlp = nn.Linear(int(hid_dim), int(hid_dim), bias=False)
        self.register_memory("y", None)
        nn.init.orthogonal_(self.mlp.weight, gain=1.0)

    def single_step_forward(self, x: torch.Tensor):
        if self.y is None:
            self.y = x.new_zeros(self.skip, x.shape[0], x.shape[1])

        if self.training:
            weights = F.softmax(self.skip_para / self.tau, dim=0)
            recurrent_spike = (weights * self.y).sum(0)
        else:
            recurrent_spike = self.y[torch.argmax(self.skip_para)]

        spike = self.sub_module(self.mlp(recurrent_spike) + x)
        self.y = torch.roll(self.y, 1, 0)
        self.y[0] = spike
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if x_seq.dim() != 3:
            raise ValueError("ASRC expects [T, N, C]")
        return torch.stack(
            [self.single_step_forward(x_seq[t]) for t in range(x_seq.shape[0])]
        )

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self.single_step_forward(x)
        return self.multi_step_forward(x)

    @torch.no_grad()
    def decrease_tau(self):
        self.tau.mul_(self.de_tau)


class AstroLIFSoma(LIFSoma , AstroSomaMixin):  ##
    """LIF soma with spike-driven astrocyte state.

    The previous astro state modulates the current membrane, while the
    current spike updates the astro state for future time steps.
    """

    def __init__(
        self, tau: float = 2., decay_input: bool = True,
        v_threshold: float = 1., v_reset: float = 0.,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = True, step_mode: str = "m",
        backend: str = "torch",
        store_v_seq: bool = False, store_v_pre_spike: bool = False,
        astro_lambda: float = 0.1, astro_trace_decay: float = 0.9,
        astro_gain: float = 0.5, astro_bias_gain: float = 0.0,
        astro_pool_kernel: int = 5, astro_pool_mode: str = "avg",  # pool_kernel
        astro_event_write: bool = True, astro_event_threshold: float = 0.5,
        astro_event_slope: float = 8.0, astro_delta_weight: float = 0.5,
        store_c_seq: bool = False,
    ):
        super().__init__(
            tau=tau, decay_input=decay_input, v_threshold=v_threshold,
            v_reset=v_reset, surrogate_function=surrogate_function,
            detach_reset=detach_reset, step_mode=step_mode, 
            backend=backend, store_v_seq=store_v_seq, store_v_pre_spike=store_v_pre_spike,
        )
        self._init_astro_state(
            astro_lambda=astro_lambda,
            astro_trace_decay=astro_trace_decay,
            astro_gain=astro_gain,
            astro_bias_gain=astro_bias_gain,
            astro_spike_scale=1.0,
            astro_pool_kernel=astro_pool_kernel,
            astro_pool_mode=astro_pool_mode,
            astro_event_write=astro_event_write,
            astro_event_threshold=astro_event_threshold,
            astro_event_slope=astro_event_slope,
            astro_delta_weight=astro_delta_weight,
            store_c_seq=store_c_seq,
        )

    def _single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.c_float_to_tensor(x)
        self.neuronal_charge(x)
        self.v = self.astro_modulate_membrane(self.v)
        v_pre_spike = self.v
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        self.astro_update(spike)
        return spike, v_pre_spike, self.c

    def single_step_forward(self, x: torch.Tensor):
        spike, v_pre_spike, _ = self._single_step_forward(x)
        if self.store_v_pre_spike:
            self.v_pre_spike = v_pre_spike
        #self.firing_rate = spike.float().mean()    
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        v_pre_spike_seq = [] if self.store_v_pre_spike else None
        v_seq = [] if self.store_v_seq else None
        c_seq = [] if self.store_c_seq else None
        for t in range(T):
            y, v_pre_spike, c = self._single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_pre_spike:
                v_pre_spike_seq.append(v_pre_spike)
            if self.store_v_seq:
                v_seq.append(self.v)
            if self.store_c_seq:
                c_seq.append(c)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        if self.store_v_pre_spike:
            self.v_pre_spike = torch.stack(v_pre_spike_seq)
        if self.store_c_seq:
            self.c_seq = torch.stack(c_seq)
        spike_seq = torch.stack(y_seq)
        self.firing_rate = spike_seq.float().mean()
        return spike_seq


class IFSoma(BaseSoma):

    def __init__(
        self, v_threshold: float = 1., v_reset: float = 0.,
        surrogate_function: Callable = surrogate.Sigmoid(), 
        detach_reset: bool = False, step_mode='s', backend='torch', 
        store_v_seq: bool = False, store_v_pre_spike: bool = False
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, 
            step_mode, backend, store_v_seq, store_v_pre_spike
        )

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    @staticmethod
    @torch.jit.script
    def jit_single_eval_hard(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float
    ):
        v_pre_spike = v + x
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = v_reset * spike + v_pre_spike * (1. - spike)
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_single_eval_soft(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float
    ):
        v_pre_spike = v + x
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = v_pre_spike - spike * v_threshold
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            return super().single_step_forward(x)
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None: # soft reset
                spike, v_pre_spike, self.v = (
                    self.jit_single_eval_soft(x, self.v, self.v_threshold)
                )
            else:
                spike, v_pre_spike, self.v = (
                    self.jit_single_eval_hard(
                        x, self.v, self.v_threshold, self.v_reset
                    )
                )
            if self.store_v_pre_spike:
                self.v_pre_spike = v_pre_spike
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.training:
            return super().multi_step_forward(x_seq)
        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None: # soft reset
                if self.store_v_seq:
                    spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                        self.jit_multi_eval_soft_v_seq(
                            x_seq, self.v, self.v_threshold
                        )
                    )
                else:
                    spike_seq, v_pre_spike_seq, self.v = (
                        self.jit_multi_eval_soft(
                            x_seq, self.v, self.v_threshold
                        )
                    )
            else:
                if self.store_v_seq:
                    spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                        self.jit_multi_eval_hard_v_seq(
                            x_seq, self.v, self.v_threshold, self.v_reset
                        )
                    )
                else:
                    spike_seq, v_pre_spike_seq, self.v = (
                        self.jit_multi_eval_hard(
                            x_seq, self.v, self.v_threshold, self.v_reset
                        )
                    )
            if self.store_v_pre_spike:
                self.v_pre_spike = v_pre_spike_seq
            return spike_seq
