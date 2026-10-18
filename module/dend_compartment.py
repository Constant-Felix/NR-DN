"""The voltage dynamics of dendritic compartments.

This package contains a series of classes depicting different types of dendritic
compartments so that we can compute the dendritic voltage dynamics step by step,
given the input to all the compartments. When computing the dendritic voltage 
dynamics for a single time step, the compartments are treated independently. 
The relationship (wiring) among a set of compartments is not considered here.
"""

import abc
from typing import Callable, Optional
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from spikingjelly.activation_based import base
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
    # Cropping an irfft result returns a view that otherwise keeps the full
    # zero-padded (often 2T) storage alive for the complete autograd graph.
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


def _causal_fft_template_convolution_impl(
    x: torch.Tensor,
    kernel_template: torch.Tensor,
    branch_index: torch.Tensor,
    input_gain: torch.Tensor,
) -> torch.Tensor:
    """Convolve with branch templates gathered after their shared FFT."""
    T = x.shape[0]
    kernel_length = kernel_template.shape[-1]
    if T <= 0 or kernel_length <= 0:
        raise ValueError("x and kernel_template must have non-empty time dimensions")

    linear_length = T + kernel_length - 1
    n_fft = 1 << (linear_length - 1).bit_length()
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    x_work = x.movedim(0, -1).to(dtype=work_dtype)
    kernel_work = kernel_template.to(device=x.device, dtype=work_dtype)

    x_f = torch.fft.rfft(x_work, n=n_fft, dim=-1)
    kernel_f = torch.fft.rfft(kernel_work, n=n_fft, dim=-1)
    kernel_f = kernel_f[branch_index.to(device=kernel_f.device)]

    spatial_dims = x.dim() - 5
    kernel_f = kernel_f.reshape(
        1,
        *kernel_f.shape[:-1],
        *([1] * spatial_dims),
        kernel_f.shape[-1],
    )
    input_gain = input_gain.to(device=x.device, dtype=work_dtype)
    y = torch.fft.irfft(
        x_f * kernel_f * input_gain,
        n=n_fft,
        dim=-1,
    )[..., :T].clone(memory_format=torch.contiguous_format)
    return y.movedim(-1, 0).to(dtype=x.dtype)


def _causal_fft_template_convolution_time_first(
    x: torch.Tensor,
    kernel_template: torch.Tensor,
    branch_index: torch.Tensor,
    input_gain: torch.Tensor,
) -> torch.Tensor:
    """Apply shared branch kernels while recomputing FFT workspaces in backward."""
    if torch.is_grad_enabled() and (
        x.requires_grad or kernel_template.requires_grad or input_gain.requires_grad
    ):
        return checkpoint(
            _causal_fft_template_convolution_impl,
            x,
            kernel_template,
            branch_index,
            input_gain,
            use_reentrant=False,
        )
    return _causal_fft_template_convolution_impl(
        x,
        kernel_template,
        branch_index,
        input_gain,
    )


class BaseDendCompartment(base.MemoryModule, abc.ABC):
    """Base class for all dendritic compartments.

    Attributes:
        v (Union[float, torch.Tensor]): voltage of the dendritic compartment(s)
            at the current time step.
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode.
        store_v_seq (bool): whether to store the compartmental potential at 
            every time step when using multi-step mode. If True, there is 
            another attribute called v_seq.
    """

    def __init__(
        self, v_init: float = 0., 
        step_mode: str = "s", store_v_seq: bool = False,
    ):
        """The constructor of BaseDendCompartment.

        Args:
            v_init (float, optional): initial voltage (at time step 0). 
                Defaults to 0..
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
            store_v_seq (bool, optional): whether to store the compartmental 
                potential at every time step when using multi-step mode. 
                Defaults to False.
        """
        super().__init__()
        self.register_memory("v", v_init)
        self.step_mode = step_mode
        self.store_v_seq = store_v_seq

    @property
    def store_v_seq(self) -> bool:
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, val: bool):
        self._store_v_seq = val
        if val and (not hasattr(self, "v_seq")):
            self.register_memory("v_seq", None)

    def v_float2tensor(self, x: torch.Tensor):
        """If self.v is a float, turn it into a tensor with x's shape.
        """
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)

    @abc.abstractmethod
    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        return torch.stack(y_seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self.single_step_forward(x)
        elif self.step_mode == "m":
            return self.multi_step_forward(x)
        else:
            raise ValueError(
                f"BaseDendCompartment.step_mode shoud be 'm' or 's', "
                f"but get {self.step_mode} instead."
            )



        
class MultiScaleDendCompartment(BaseDendCompartment):
    """
    Passive dendritic compartment with learnable SCALAR tau.
    
    Fixed:
    1. Causal Matrix Construction (Lower Triangular).
    2. Supports input shape (T, B, C_sub, H, W, N).
    3. Tau is shared across all dimensions.
    """

    def __init__(
        self,
        num_branches,
        c_sub=1,
        init_tau = [1.25,10.0], 
        soma_dim = 3,
        decay_input: bool = True,  ###
        gating = True,
        bn = True,  ##
        res = False,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter = False,
        use_astro: bool = True, 
        last_sigmoid = False,
        last_tanh = True,
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        self.soma_dim = soma_dim
        self.gating = gating
        self.bn = bn
        #self.bn_alter = bn_alter
        self.skip_weight = nn.Parameter(torch.tensor(0.01))
        self.res = res
        self.no_filter = no_filter
        self.use_astro = use_astro
        self.last_sigmoid = last_sigmoid
        self.last_tanh = last_tanh
        #self.use_oli = use_oli
        self.c_sub = c_sub
        self.num_branches = num_branches
        self.astro_bias = nn.Parameter(torch.full((1, 1, 1, 1, num_branches), -1.0))
        #self.alpha = nn.Parameter(torch.tensor(0.7))
        #self.beta = nn.Parameter(torch.tensor(1.0))
        #self.w_b = nn.Parameter(torch.tensor(0.1))
        if init_tau is None:
            tau_data = torch.linspace(2.0, 10.0, steps=num_branches)
        elif isinstance(init_tau, (float, int)):
            tau_data = torch.full((num_branches,), float(init_tau))
        else:
            tau_data = torch.as_tensor(init_tau, dtype=torch.float32)

        self.tau_branches = nn.Parameter(tau_data.clone().detach().float())    
        self.decay_input = decay_input
        self.v_rest = v_rest

        #if self.use_oli:
                # [少突胶质细胞] 动态 Tau 调节系数
                # 髓鞘化灵敏度，初始化为 0.5
            #self.oligo_alpha = nn.Parameter(torch.tensor(0.5))

        if self.use_astro:
                # [星形胶质细胞] 慢速积分状态
                # 钙离子衰减系数 (0.9 ~ 0.99)
            self.astro_decay = nn.Parameter(torch.tensor(0.9))
                # 钙离子对门控的影响力 (Gain)
            self.astro_gain = nn.Parameter(torch.tensor(1.0))
                # 内部状态
            self.ca_state = 0.0 

        if self.gating:
            if self.bn:
                if last_sigmoid == True or last_tanh == True:
                    if self.soma_dim == 3:
                    # BN + Gating 模式 (静态图/DVS通用)
                        self.gate_alpha = nn.Parameter(torch.ones(1, 1, 1, 1, 1, num_branches))
                        self.gate_beta = nn.Parameter(torch.zeros(1, 1, 1, 1, 1, num_branches)) #if last_sigmoid == True else nn.Parameter(torch.full((1, 1, 1, 1, 1, num_branches), -1.0))
                    else:
                        self.gate_alpha = nn.Parameter(torch.ones(1, 1, 1, 1, num_branches))
                        self.gate_beta = nn.Parameter(torch.zeros(1, 1, 1, 1, num_branches)) #if last_sigmoid == True else nn.Parameter(torch.full((1, 1, 1, 1, num_branches), -1.0))

                self.branch_bn = nn.BatchNorm2d(self.c_sub, affine=True, momentum=0.1) if self.soma_dim == 3 else nn.BatchNorm1d(self.c_sub, affine=True, momentum=0.1)  
                nn.init.constant_(self.branch_bn.weight, 1.0) ## 1.5
                nn.init.constant_(self.branch_bn.bias, 0.0) ## 0.5
            else:
                if last_sigmoid == True or last_tanh == True:
                    self.gate_scale = nn.Parameter(torch.tensor(1.0))
                    self.gate_beta2 = nn.Parameter(torch.tensor(0.0))
        
    def _build_tau_terms(self, T: int, device, dtype):
        """
        现在 tau_branches 是 (N,) 的张量。
        为了能使用矩阵乘法，我们要对 N 个分支分别构建积分矩阵。
        """
        # 限制最小 tau，防止衰减为 0
        tau = torch.clamp(self.tau_branches, min=1.0 + 1e-5) # 形状: (N,)
        a = 1.0 - 1.0 / tau # 形状: (N,)

        t = torch.arange(T, device=device, dtype=dtype)
        i = t[:, None]
        j = t[None, :]
        diff = i - j
        mask = diff >= 0
        
        # 构建 tau_matrix，形状将会是 (N, T, T)
        # 这意味着 N 个分支各有自己的一套下三角积分矩阵
        tau_matrix = torch.zeros((len(tau), T, T), device=device, dtype=dtype)
        
        for n in range(len(tau)):
            tau_matrix[n] = tau_matrix[n].masked_scatter(mask, (a[n] ** diff[mask]).to(dtype))

        # 系数 v_init 和 v_rest 也会变成 (N, T)
        tau_vec_init = a.unsqueeze(1) ** (t + 1.0) # (N, T)
        tau_vec_rest = 1.0 - tau_vec_init # (N, T)

        return tau_matrix, tau_vec_init, tau_vec_rest, tau
    
    
    def _compute_astro_modulation(self, y_seq):
        """
        计算星形胶质细胞的钙信号，独立调节每个分支 (N) 的强度。
        """
        if not self.use_astro:
            return 1.0 # 无调节
            
        T = y_seq.shape[0]
        B = y_seq.shape[1]
        N = y_seq.shape[-1]  # [关键修改 1]: 动态获取分支数 N
        device = y_seq.device
        
        # 1. 能量输入 (取绝对值)
        # [关键修改 2]: 去掉在维度 N 上的均值操作

        if self.soma_dim == 3:
            # y_seq: (T, B, C, H, W, N) -> 对 C(2), H(3), W(4) 求均值，保留 N(5)
            energy_seq = y_seq.abs().mean(dim=(2, 3, 4)) # 结果 shape: (T, B, N)
        else:
            # y_seq: (T, B, C, H, N) -> 对 C(2), H(3) 求均值，保留 N(4)
            energy_seq = y_seq.abs().mean(dim=(2, 3))    # 结果 shape: (T, B, N)
        
        # 2. 慢速积分 (Leaky Integration)
        ca_seq = []
        
        # [关键修改 3]: 初始化时给每个样本的每个分支分配独立的钙状态
        curr_ca = torch.zeros((B, N), device=device)

        decay = torch.clamp(self.astro_decay, 0.0, 1.0)
        
        for t in range(T):
            curr_ca = decay * curr_ca + (1 - decay) * energy_seq[t]  ## 加了tanh
            ca_seq.append(curr_ca)
            
        # 更新状态
        self.ca_state = curr_ca.detach()
         
        # 3. 生成调节系数 (Modulation Factor)
        # ca_signal 的 shape 是 (T, B, N)
        ca_signal = torch.stack(ca_seq, dim=0)
        
        # [关键修改 4]: 巧妙重塑形状，让 Pytorch 的广播机制发挥作用
        if self.soma_dim == 3:
            # 变回 (T, B, 1, 1, 1, N) 从而能与 (T, B, C, H, W, N) 完美相乘
            ca_signal = ca_signal.view(T, B, 1, 1, 1, N)
        else:
            # 变回 (T, B, 1, 1, N) 从而能与 (T, B, C, H, N) 完美相乘
            ca_signal = ca_signal.view(T, B, 1, 1, N)    
        
        #ca_baseline = ca_signal.mean(dim=(0, 1), keepdim=True) 
        #modulation = 1.0 + torch.tanh(self.astro_gain * (ca_signal - ca_baseline))
        #add = torch.tanh(self.w_b * ca_signal) # (B, N)
        modulation = 1.0 + 0.5 * torch.tanh(self.astro_gain * ca_signal + self.astro_bias)

        return modulation

    def single_step_forward(self, x: torch.Tensor):
        tau = torch.clamp(self.tau_branches, min=1.0 + 1e-3)

        if isinstance(self.v, float):
            self.v = torch.full_like(x, self.v)

        if self.decay_input:
            v = self.single_step_decay_input(self.v, x, self.v_rest, tau)
        else:
            v = self.single_step_not_decay_input(self.v, x, self.v_rest, tau)

        # IMPORTANT:
        # detach state to avoid cross-iteration graph leakage
        self.v = v.detach()
        return v

    def multi_step_forward(self, x_seq: torch.Tensor):
        """
        Args:
            x_seq: Shape (T, B, C_sub, H, W, N)
        Returns:
            y: Shape (T, B, C_sub, H, W, N)
        """
        if self.soma_dim==3:
            T, B, C_sub, H, W, N = x_seq.shape # T,B,C/N,H,W,N
        else:
            T, B, C_sub, H, N = x_seq.shape  # W,B,C/N,H,N
        device = x_seq.device
        dtype = x_seq.dtype
        
        if self.no_filter == False:

        
            (tau_matrix, tau_vec_init, tau_vec_rest, tau) = \
            self._build_tau_terms(T, device, dtype)

            if self.decay_input:
                
                if self.soma_dim == 3:
                    div_tau = tau.view(1, 1, 1, 1, 1, N)
                else:
                    div_tau = tau.view(1, 1, 1, 1, N)    
                x_seq = x_seq / div_tau

            y_branches = []
            for n in range(self.num_branches):
                x_n = x_seq[..., n].reshape(T, -1)
                y_n_flat = torch.matmul(tau_matrix[n], x_n)
                if self.soma_dim == 3:
                    y_n = y_n_flat.view(T, B, C_sub, H, W)
                else:
                    y_n = y_n_flat.view(T, B, C_sub, H)    
                
                # Init & Rest
                t_init = tau_vec_init[n].view(T, 1, 1, 1, 1) if self.soma_dim == 3 else tau_vec_init[n].view(T, 1, 1, 1)
                t_rest = tau_vec_rest[n].view(T, 1, 1, 1, 1) if self.soma_dim == 3 else tau_vec_rest[n].view(T, 1, 1, 1)
                
                v_init = self.v
                if isinstance(v_init, torch.Tensor):
                    v_in = v_init[..., n].detach().unsqueeze(0)
                else:
                    v_in = v_init
                    
                y_n = y_n + (t_init * v_in) + (t_rest * self.v_rest)
                y_branches.append(y_n)

            y = torch.stack(y_branches, dim=-1) # (T, B, ..., N)
            
        
        else:
            y = x_seq
              
        if self.store_v_seq:
            self.v_seq = y

        self.v = y[-1].detach()
        
        if self.gating:

            m_mod = self._compute_astro_modulation(y)
            y = y * m_mod
            if self.bn:
                
                
                if self.soma_dim == 3:
                        y_permuted = y.permute(0, 1, 5, 2, 3, 4).contiguous()
                        y_reshaped = y_permuted.view(T * B, N * C_sub, H, W)
                        
                        y_normed_reshaped = self.branch_bn(y_reshaped)
                        
                        y_normed = y_normed_reshaped.view(T, B, N, C_sub, H, W)

                        y_normed = y_normed.permute(0, 1, 3, 4, 5, 2).contiguous()

                else:
                        y_permuted = y.permute(0, 1, 4, 2, 3).contiguous()
                        
                        # 2. 融合维度：(Batch, Channels, Length)
                        # 目标形状: (T * B, C * N, H)
                        y_reshaped = y_permuted.view(T * B, N * C_sub, H)
                        
                        # 3. 过 BatchNorm1d
                        y_normed_reshaped = self.branch_bn(y_reshaped)
                        
                        # 4. 还原形状：先解开 view -> (T, B, C, N, H)
                        y_normed = y_normed_reshaped.view(T, B, N, C_sub, H)
                        
                        # 5. 再把 N 移回最后 -> (T, B, C, H, N)
                        y_normed = y_normed.permute(0, 1, 3, 4, 2).contiguous()        

                #m_mod = self._compute_astro_modulation(y_normed)
                y_out = y_normed
                if self.last_sigmoid == True:
                    gate = torch.sigmoid(self.gate_alpha * y_out + self.gate_beta)  ##这里用的y_out,而不是y？
                    y_out = y_out * gate
                if self.last_tanh == True:
                    gate = torch.tanh(self.gate_alpha * y_out + self.gate_beta)  ##这里用的y_out,而不是y？
                    y_out = y_out * gate   
                
            else:
                y_out = y
                if self.last_sigmoid == True:
                    gate_factor = torch.sigmoid(self.gate_scale * y_out + self.gate_beta2)
                    y_out = y_out * gate_factor
                if self.last_tanh == True:
                    gate_factor = torch.tanh(self.gate_scale * y_out + self.gate_beta2)
                    y_out = y_out * gate_factor        
        else:
            y_out = y


        if self.res == True:
            y_out = y_out + self.skip_weight * x_seq
          
               
            
        return y_out


class SparseRoutedTrunkDistalDendCompartment(BaseDendCompartment):
    """Sparse-routed trunk-distal dendrite for branch scale-up.

    The input compartment dimension is branch-major:
    ``[b0_k0, b0_k1, ..., b1_k0, b1_k1, ...]``.  In every branch, compartment
    ``k0`` is the trunk that carries the signed main signal.  Distal
    compartments form a bounded nonlinear context that modulates the trunk.

    This class deliberately avoids the old BN + ``x * sigmoid(x)`` path.  It is
    designed to make increasing ``num_branches`` add candidate dendritic routes
    rather than unconditionally adding more branch outputs to the soma.
    """

    def __init__(
        self,
        num_branches,
        compartments_per_branch=2,
        c_sub=1,
        init_tau=None,
        tau_min: float = 1.25,
        tau_max: float = 8.0,
        compartment_tau_scale=(1.0, 1.5),
        soma_dim=3,
        decay_input: bool = True,
        selective_update: bool = True,
        active_branches: int = 2,
        route_temperature: float = 1.0,
        route_with_update_gate: bool = True,
        residual_scale: float = 0.1,
        distal_gain_init: float = 0.1,
        update_threshold: float = 0.5,
        update_slope: float = 8.0,
        update_delta_weight: float = 0.5,
        update_activity_weight: float = 0.5,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter=False,
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if compartments_per_branch < 2:
            raise ValueError("compartments_per_branch must be at least 2")
        if active_branches <= 0:
            raise ValueError("active_branches must be positive")
        if route_temperature <= 0:
            raise ValueError("route_temperature must be positive")
        if tau_min <= 1.0 or tau_max <= 1.0:
            raise ValueError("tau_min and tau_max should be larger than 1")

        self.soma_dim = soma_dim
        self.c_sub = c_sub
        self.num_branches = int(num_branches)
        self.compartments_per_branch = int(compartments_per_branch)
        self.distal_count = self.compartments_per_branch - 1
        self.num_compartments = self.num_branches * self.compartments_per_branch
        self.decay_input = decay_input
        self.selective_update = bool(selective_update)
        self.active_branches = min(int(active_branches), self.num_branches)
        self.route_temperature = float(route_temperature)
        self.route_with_update_gate = bool(route_with_update_gate)
        self.no_filter = no_filter
        self.v_rest = v_rest

        tau_data = self._make_tau_data(
            init_tau=init_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            compartment_tau_scale=compartment_tau_scale,
        )
        self.tau_compartments = nn.Parameter(tau_data.clone().detach().float())

        inv_softplus_one = torch.log(torch.expm1(torch.tensor(1.0)))
        self.distal_shift = nn.Parameter(torch.zeros(self.num_branches, self.distal_count))
        self.distal_log_scale = nn.Parameter(torch.full((self.num_branches, self.distal_count), float(inv_softplus_one)))
        self.distal_gain = nn.Parameter(torch.full((self.num_branches,), float(distal_gain_init)))
        self.branch_strength = nn.Parameter(torch.ones(self.num_branches))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

        self.route_bias = nn.Parameter(torch.zeros(self.num_branches))
        self.route_score_scale = nn.Parameter(torch.tensor(1.0))
        self.route_update_gain = nn.Parameter(torch.tensor(0.5))

        self.update_threshold = nn.Parameter(torch.full((self.num_branches,), float(update_threshold)))
        self.update_delta_weight = nn.Parameter(torch.tensor(float(update_delta_weight)))
        self.update_activity_weight = nn.Parameter(torch.tensor(float(update_activity_weight)))
        self.update_slope = float(update_slope)

        self.update_gate_seq = None
        self.route_seq = None
        self.distal_energy_seq = None
        self.branch_output_seq = None

    def _make_tau_data(self, init_tau, tau_min, tau_max, compartment_tau_scale):
        if init_tau is not None:
            tau_data = torch.as_tensor(init_tau, dtype=torch.float32)
            if tau_data.numel() == 1:
                return torch.full((self.num_compartments,), float(tau_data.flatten()[0]))
            if tau_data.numel() == 2:
                tau_min_v = float(tau_data.flatten()[0])
                tau_max_v = float(tau_data.flatten()[-1])
                branch_tau = torch.exp(torch.linspace(
                    torch.log(torch.tensor(tau_min_v)),
                    torch.log(torch.tensor(tau_max_v)),
                    steps=self.num_branches,
                ))
                comp_scale = torch.linspace(1.0, 1.5, steps=self.compartments_per_branch)
                return (branch_tau[:, None] * comp_scale[None, :]).reshape(-1)
            if tau_data.numel() == self.num_branches:
                return tau_data.flatten().repeat_interleave(self.compartments_per_branch)
            if tau_data.numel() == self.compartments_per_branch:
                return tau_data.flatten().repeat(self.num_branches)
            if tau_data.numel() == self.num_compartments:
                return tau_data.reshape(-1)
            raise ValueError(
                "init_tau must be scalar, length 2, length num_branches, "
                "length compartments_per_branch, or length num_branches * compartments_per_branch"
            )

        branch_tau = torch.exp(torch.linspace(
            torch.log(torch.tensor(float(tau_min))),
            torch.log(torch.tensor(float(tau_max))),
            steps=self.num_branches,
        ))
        if isinstance(compartment_tau_scale, (float, int)):
            comp_scale = torch.full((self.compartments_per_branch,), float(compartment_tau_scale))
        else:
            comp_scale_data = torch.as_tensor(compartment_tau_scale, dtype=torch.float32)
            if comp_scale_data.numel() == 2:
                comp_scale = torch.linspace(
                    float(comp_scale_data.flatten()[0]),
                    float(comp_scale_data.flatten()[-1]),
                    steps=self.compartments_per_branch,
                )
            elif comp_scale_data.numel() == self.compartments_per_branch:
                comp_scale = comp_scale_data.flatten()
            else:
                raise ValueError(
                    "compartment_tau_scale must be scalar, length 2, "
                    "or length compartments_per_branch"
                )
        return (branch_tau[:, None] * comp_scale[None, :]).reshape(-1)

    def _assert_input_shape(self, x: torch.Tensor):
        if x.shape[-1] != self.num_compartments:
            raise ValueError(
                f"Expected last input dimension to be {self.num_compartments} "
                f"(num_branches * compartments_per_branch), but got {x.shape[-1]}"
            )

    def _reshape_to_branches(self, x: torch.Tensor):
        self._assert_input_shape(x)
        return x.reshape(*x.shape[:-1], self.num_branches, self.compartments_per_branch)

    def _view_branch_param(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(*([1] * (ref.dim() - 1)), self.num_branches)

    def _view_distal_param(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(*([1] * (ref.dim() - 2)), self.num_branches, self.distal_count)

    def _compute_update_gate(self, x: torch.Tensor, v_prev: torch.Tensor):
        branch_input = self._reshape_to_branches(x)
        branch_state = self._reshape_to_branches(v_prev)
        if not self.selective_update:
            return torch.ones_like(branch_input[..., 0])

        activity = branch_input.abs().mean(dim=-1)
        delta = (branch_input - branch_state).abs().mean(dim=-1)
        score = self.update_activity_weight.abs() * activity + self.update_delta_weight.abs() * delta
        threshold = self._view_branch_param(self.update_threshold.abs(), score)
        return torch.sigmoid(self.update_slope * (score - threshold))

    def _update_compartment_state(self, x: torch.Tensor):
        self._assert_input_shape(x)
        if isinstance(self.v, float):
            self.v = torch.full_like(x, self.v)

        v_prev = self.v
        update_gate = self._compute_update_gate(x, v_prev)
        gate = update_gate.unsqueeze(-1).expand(*update_gate.shape, self.compartments_per_branch)
        gate = gate.reshape_as(x)

        if self.no_filter:
            candidate = x
        else:
            tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-3)
            tau = tau.view(*([1] * (x.dim() - 1)), self.num_compartments)
            if self.decay_input:
                candidate = v_prev + (x - (v_prev - self.v_rest)) / tau
            else:
                candidate = v_prev - (v_prev - self.v_rest) / tau + x

        v = (1.0 - gate) * v_prev + gate * candidate
        self.v = v.detach()
        self.update_gate_seq = update_gate
        return v, update_gate

    def _mexican_hat(self, energy: torch.Tensor):
        energy_sq = energy.square()
        return (1.0 - energy_sq) * torch.exp(-0.5 * energy_sq)

    def _compute_branch_output(self, state: torch.Tensor, raw_input: torch.Tensor):
        branch_state = self._reshape_to_branches(state)
        branch_input = self._reshape_to_branches(raw_input)
        trunk = branch_state[..., 0]
        distal = branch_state[..., 1:]

        scale = F.softplus(self._view_distal_param(self.distal_log_scale, distal)) + 1e-4
        shift = self._view_distal_param(self.distal_shift, distal)
        distal_norm = (distal - shift) / scale
        distal_energy = torch.sqrt(distal_norm.square().mean(dim=-1) + 1e-6)
        distal_wave = self._mexican_hat(distal_energy)

        modulation = 1.0 + self._view_branch_param(self.distal_gain, trunk) * distal_wave
        residual = self.residual_scale * branch_input.mean(dim=-1)
        y = self._view_branch_param(self.branch_strength, trunk) * trunk * modulation + residual

        self.distal_energy_seq = distal_energy
        self.branch_output_seq = y
        return y

    def _compute_route(self, branch_output: torch.Tensor, update_gate: torch.Tensor):
        logits = self.route_score_scale.abs() * branch_output.abs()
        logits = logits + self._view_branch_param(self.route_bias, branch_output)
        if self.route_with_update_gate:
            logits = logits + self.route_update_gain * update_gate

        probs = torch.softmax(logits / self.route_temperature, dim=-1)
        if self.active_branches >= self.num_branches:
            route = probs
        else:
            topk_idx = torch.topk(logits, k=self.active_branches, dim=-1).indices
            mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
            route = probs * mask
            route = route / route.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.route_seq = route.detach()
        return route

    def single_step_forward(self, x: torch.Tensor):
        state, update_gate = self._update_compartment_state(x)
        y = self._compute_branch_output(state, x)
        route = self._compute_route(y, update_gate)
        return y * route

    def multi_step_forward(self, x_seq: torch.Tensor):
        self._assert_input_shape(x_seq)
        y_seq = []
        update_gate_seq = []
        route_seq = []
        distal_energy_seq = []
        branch_output_seq = []
        v_seq = [] if self.store_v_seq else None

        for t in range(x_seq.shape[0]):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            update_gate_seq.append(self.update_gate_seq)
            route_seq.append(self.route_seq)
            distal_energy_seq.append(self.distal_energy_seq)
            branch_output_seq.append(self.branch_output_seq)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        self.update_gate_seq = torch.stack(update_gate_seq)
        self.route_seq = torch.stack(route_seq)
        self.distal_energy_seq = torch.stack(distal_energy_seq)
        self.branch_output_seq = torch.stack(branch_output_seq)
        return torch.stack(y_seq)

class ChannelPreservingTrunkDistalDendCompartment(BaseDendCompartment):
    """Channel-preserving trunk-distal dendrite.

    This module is intended for the ``C_out -> C_out`` dendrite design.  Unlike
    ``SegregatedDend``-style compartments, it does not require the preceding
    Conv/Linear layer to emit ``C * num_branches * compartments_per_branch``
    channels.  The public input and output shapes are the same:

    - single-step 1D: ``[N, C, L] -> [N, C, L]``
    - single-step 2D: ``[N, C, H, W] -> [N, C, H, W]``
    - multi-step 1D: ``[T, N, C, L] -> [T, N, C, L]``
    - multi-step 2D: ``[T, N, C, H, W] -> [T, N, C, H, W]``

    Branches and compartments are internal dimensions.  Each channel is assigned
    to a small number of branches with an overlapping fixed mask, then each
    branch keeps ``K`` compartment states.  The branch readout uses a trunk signal
    modulated by distal compartment differences, rather than a linear sum over
    compartments.
    """

    def __init__(
        self,
        channels=None,
        num_branches=4,
        compartments_per_branch=2,
        c_sub=None,
        branch_degree=2,
        init_tau=None,
        tau_min: float = 1.25,
        tau_max: float = 8.0,
        compartment_tau_scale=(1.0, 1.5),
        decay_input: bool = True,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter: bool = False,
        merge_norm: str = "sqrt",
        learn_channel_gate: bool = True,
        channel_gate_scale: float = 0.5,
        distal_gain_init: float = 0.1,
        distal_residual_init: float = 0.0,
        distal_gate_alpha_init: float = 6.0,
        distal_gate_threshold_init: float = 0.5,
        input_residual_init: float = 0.0,
        detach_state_during_forward: bool = False,
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        if channels is None:
            channels = c_sub
        if channels is None:
            raise ValueError("channels or c_sub must be provided")
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if compartments_per_branch < 2:
            raise ValueError("compartments_per_branch must be at least 2")
        if branch_degree <= 0:
            raise ValueError("branch_degree must be positive")
        if merge_norm not in {"sqrt", "mean", "sum"}:
            raise ValueError("merge_norm must be one of: 'sqrt', 'mean', 'sum'")
        if tau_min <= 1.0 or tau_max <= 1.0:
            raise ValueError("tau_min and tau_max should be larger than 1")

        self.channels = int(channels)
        self.c_sub = self.channels
        self.num_branches = int(num_branches)
        self.compartments_per_branch = int(compartments_per_branch)
        self.distal_count = self.compartments_per_branch - 1
        self.branch_degree = min(int(branch_degree), self.num_branches)
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.no_filter = no_filter
        self.merge_norm = merge_norm
        self.learn_channel_gate = bool(learn_channel_gate)
        self.channel_gate_scale = float(channel_gate_scale)
        self.detach_state_during_forward = bool(detach_state_during_forward)

        branch_mask = self._make_branch_mask(self.channels, self.num_branches, self.branch_degree)
        channel_degree = branch_mask.sum(dim=0).clamp_min(1.0)
        self.register_buffer("branch_mask", branch_mask)
        self.register_buffer("channel_degree", channel_degree)

        tau_data = self._make_tau_data(
            init_tau=init_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            compartment_tau_scale=compartment_tau_scale,
        )
        self.tau_compartments = nn.Parameter(tau_data.clone().detach().float())

        self.branch_channel_gain = nn.Parameter(torch.zeros(self.num_branches, self.channels))
        self.compartment_input_gain = nn.Parameter(torch.zeros(self.num_branches, self.compartments_per_branch))
        self.distal_mix_logits = nn.Parameter(torch.zeros(self.num_branches, self.distal_count))
        self.distal_gate_alpha = nn.Parameter(torch.full((self.num_branches, self.distal_count), float(distal_gate_alpha_init)))
        self.distal_gate_threshold = nn.Parameter(torch.full((self.num_branches, self.distal_count), float(distal_gate_threshold_init)))
        self.distal_gain = nn.Parameter(torch.full((self.num_branches,), float(distal_gain_init)))
        self.distal_residual_gain = nn.Parameter(torch.full((self.num_branches,), float(distal_residual_init)))
        self.branch_strength = nn.Parameter(torch.ones(self.num_branches))
        self.input_residual_scale = nn.Parameter(torch.tensor(float(input_residual_init)))
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self.branch_mod_seq = None
        self.distal_gate_seq = None
        self.branch_output_seq = None
        self.branch_input_seq = None

    @staticmethod
    def _make_branch_mask(channels: int, num_branches: int, branch_degree: int):
        mask = torch.zeros(num_branches, channels, dtype=torch.float32)
        channel_index = torch.arange(channels)
        for offset in range(branch_degree):
            branch_index = (channel_index + offset) % num_branches
            mask[branch_index, channel_index] = 1.0
        return mask

    def _make_tau_data(self, init_tau, tau_min, tau_max, compartment_tau_scale):
        if init_tau is not None:
            tau_data = torch.as_tensor(init_tau, dtype=torch.float32)
            if tau_data.numel() == 1:
                return torch.full((self.num_branches, self.compartments_per_branch), float(tau_data.flatten()[0]))
            if tau_data.numel() == 2:
                tau_min_v = float(tau_data.flatten()[0])
                tau_max_v = float(tau_data.flatten()[-1])
                branch_tau = torch.exp(torch.linspace(
                    torch.log(torch.tensor(tau_min_v)),
                    torch.log(torch.tensor(tau_max_v)),
                    steps=self.num_branches,
                ))
                comp_scale = torch.linspace(1.0, 1.5, steps=self.compartments_per_branch)
                return branch_tau[:, None] * comp_scale[None, :]
            if tau_data.numel() == self.num_branches:
                return tau_data.flatten()[:, None].repeat(1, self.compartments_per_branch)
            if tau_data.numel() == self.compartments_per_branch:
                return tau_data.flatten()[None, :].repeat(self.num_branches, 1)
            if tau_data.numel() == self.num_branches * self.compartments_per_branch:
                return tau_data.reshape(self.num_branches, self.compartments_per_branch)
            raise ValueError(
                "init_tau must be scalar, length 2, length num_branches, "
                "length compartments_per_branch, or length num_branches * compartments_per_branch"
            )

        branch_tau = torch.exp(torch.linspace(
            torch.log(torch.tensor(float(tau_min))),
            torch.log(torch.tensor(float(tau_max))),
            steps=self.num_branches,
        ))
        if isinstance(compartment_tau_scale, (float, int)):
            comp_scale = torch.full((self.compartments_per_branch,), float(compartment_tau_scale))
        else:
            comp_scale_data = torch.as_tensor(compartment_tau_scale, dtype=torch.float32)
            if comp_scale_data.numel() == 2:
                comp_scale = torch.linspace(
                    float(comp_scale_data.flatten()[0]),
                    float(comp_scale_data.flatten()[-1]),
                    steps=self.compartments_per_branch,
                )
            elif comp_scale_data.numel() == self.compartments_per_branch:
                comp_scale = comp_scale_data.flatten()
            else:
                raise ValueError(
                    "compartment_tau_scale must be scalar, length 2, "
                    "or length compartments_per_branch"
                )
        return branch_tau[:, None] * comp_scale[None, :]

    def _assert_input_shape(self, x: torch.Tensor):
        if x.dim() < 2:
            raise ValueError("Expected input shape [N, C, ...]")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, but got {x.shape[1]}")

    def _view_branch_channel(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.num_branches, self.channels, 1, *([1] * (ref.dim() - 4)))

    def _view_branch_compartment(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.num_branches, 1, self.compartments_per_branch, *([1] * (ref.dim() - 4)))

    def _view_distal_param(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.num_branches, 1, self.distal_count, *([1] * (ref.dim() - 4)))

    def _view_branch_param(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.num_branches, 1, *([1] * (ref.dim() - 3)))

    def _view_channel_degree(self, ref: torch.Tensor):
        return self.channel_degree.to(dtype=ref.dtype, device=ref.device).view(
            1, self.channels, *([1] * (ref.dim() - 2))
        )

    def _branch_channel_weight(self, dtype, device):
        weight = self.branch_mask.to(dtype=dtype, device=device)
        if self.learn_channel_gate:
            gate = 1.0 + self.channel_gate_scale * torch.tanh(self.branch_channel_gain.to(dtype=dtype, device=device))
            weight = weight * gate
        return weight

    def _build_branch_input(self, x: torch.Tensor):
        self._assert_input_shape(x)
        x_branch = x.unsqueeze(1).unsqueeze(3)
        branch_weight = self._branch_channel_weight(x.dtype, x.device)
        branch_weight = self._view_branch_channel(branch_weight, x_branch)
        comp_gain = 1.0 + 0.5 * torch.tanh(self.compartment_input_gain.to(dtype=x.dtype, device=x.device))
        comp_gain = self._view_branch_compartment(comp_gain, x_branch)
        branch_input = x_branch * branch_weight * comp_gain
        self.branch_input_seq = branch_input.detach()
        return branch_input

    def _init_state(self, branch_input: torch.Tensor):
        if isinstance(self.v, float):
            return torch.full_like(branch_input, self.v)
        if self.v.shape != branch_input.shape:
            return torch.full_like(branch_input, self.v_rest)
        return self.v

    def _integrate(self, branch_input: torch.Tensor, v_prev: torch.Tensor):
        if self.no_filter:
            return branch_input

        tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-3)
        tau = self._view_branch_compartment(tau.to(dtype=branch_input.dtype, device=branch_input.device), branch_input)
        if self.decay_input:
            return v_prev + (branch_input - (v_prev - self.v_rest)) / tau
        return v_prev - (v_prev - self.v_rest) / tau + branch_input

    def _normalize_distal_delta(self, delta: torch.Tensor):
        reduce_dims = [2] + list(range(4, delta.dim()))
        rms = torch.sqrt(delta.square().mean(dim=reduce_dims, keepdim=True) + 1e-6)
        return delta / rms

    def _compute_branch_output(self, state: torch.Tensor):
        trunk = state[:, :, :, 0, ...]
        distal = state[:, :, :, 1:, ...]
        delta = distal - trunk.unsqueeze(3)
        delta_norm = self._normalize_distal_delta(delta)

        alpha = F.softplus(self._view_distal_param(self.distal_gate_alpha, delta_norm))
        threshold = self._view_distal_param(self.distal_gate_threshold.abs(), delta_norm)
        distal_gate = torch.sigmoid(alpha * delta_norm.abs() - threshold)

        mix = torch.softmax(self.distal_mix_logits, dim=-1)
        mix = self._view_distal_param(mix.to(dtype=state.dtype, device=state.device), delta_norm)
        branch_mod = (mix * distal_gate * torch.tanh(delta_norm)).sum(dim=3)
        branch_mod = torch.tanh(branch_mod)

        distal_gain = self._view_branch_param(self.distal_gain, trunk)
        residual_gain = self._view_branch_param(self.distal_residual_gain, trunk)
        branch_strength = self._view_branch_param(self.branch_strength, trunk)
        y = branch_strength * (trunk + distal_gain * trunk * branch_mod + residual_gain * branch_mod)

        self.branch_mod_seq = branch_mod.detach()
        self.distal_gate_seq = distal_gate.detach()
        self.branch_output_seq = y.detach()
        return y

    def _merge_branches(self, branch_output: torch.Tensor, raw_input: torch.Tensor):
        y = branch_output.sum(dim=1)
        degree = self._view_channel_degree(y)
        if self.merge_norm == "sqrt":
            y = y / torch.sqrt(degree)
        elif self.merge_norm == "mean":
            y = y / degree
        y = self.output_scale * y
        return y + self.input_residual_scale * raw_input

    def _step(self, x: torch.Tensor, v_prev: torch.Tensor):
        branch_input = self._build_branch_input(x)
        state = self._integrate(branch_input, v_prev)
        branch_output = self._compute_branch_output(state)
        y = self._merge_branches(branch_output, x)
        return y, state

    def single_step_forward(self, x: torch.Tensor):
        branch_input = self._build_branch_input(x)
        v_prev = self._init_state(branch_input)
        state = self._integrate(branch_input, v_prev)
        branch_output = self._compute_branch_output(state)
        y = self._merge_branches(branch_output, x)
        self.v = state.detach()
        if self.store_v_seq:
            self.v_seq = state.detach().unsqueeze(0)
        return y

    def multi_step_forward(self, x_seq: torch.Tensor):
        if x_seq.dim() < 3:
            raise ValueError("Expected multi-step input shape [T, N, C, ...]")
        if x_seq.shape[2] != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, but got {x_seq.shape[2]}")

        y_seq = []
        v_seq = [] if self.store_v_seq else None
        branch_mod_seq = []
        distal_gate_seq = []
        branch_output_seq = []

        first_branch_input = self._build_branch_input(x_seq[0])
        v = self._init_state(first_branch_input)
        for t in range(x_seq.shape[0]):
            if t == 0:
                branch_input = first_branch_input
                state = self._integrate(branch_input, v)
                branch_output = self._compute_branch_output(state)
                y = self._merge_branches(branch_output, x_seq[t])
            else:
                y, state = self._step(x_seq[t], v)
            y_seq.append(y)
            branch_mod_seq.append(self.branch_mod_seq)
            distal_gate_seq.append(self.distal_gate_seq)
            branch_output_seq.append(self.branch_output_seq)
            if self.store_v_seq:
                v_seq.append(state.detach())
            v = state.detach() if self.detach_state_during_forward else state

        self.v = v.detach()
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        self.branch_mod_seq = torch.stack(branch_mod_seq)
        self.distal_gate_seq = torch.stack(distal_gate_seq)
        self.branch_output_seq = torch.stack(branch_output_seq)
        return torch.stack(y_seq)

class SparseChannelPreservingTrunkDistalDendCompartment(BaseDendCompartment):
    """Active-edge channel-preserving trunk-distal dendrite.

    This is the sparse version of ``ChannelPreservingTrunkDistalDendCompartment``.
    The external mapping is still ``C -> C``, but the internal state is
    ``[N, d, C, K, ...]`` instead of ``[N, B, C, K, ...]``.  Here ``B`` is the
    size of the branch parameter library and ``d`` is ``branch_degree``.  Scaling
    ``num_branches`` therefore adds candidate branch dynamics without forcing
    every channel to materialize every branch.

    Set ``free_window_order`` to a positive integer ``P`` to replace the
    exponential compartment integration with the causal sliding window used by
    ``MaskedSlidingPSN``. Each branch owns one learnable vector of length ``P``
    and all compartments routed to that branch share it. ``integration_backend``
    selects the original dense GEMM or an equivalent FFT convolution.
    """

    def __init__(
        self,
        channels=None,
        num_branches=4,
        compartments_per_branch=2,
        c_sub=None,
        branch_degree=2,
        branch_assignment: str = "cyclic",
        init_tau=None,
        tau_min: float = 1.25,
        tau_max: float = 5.0,
        compartment_tau_scale=(1.0, 2.5),
        decay_input: bool = True,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        store_branch_monitor: bool = False,
        no_filter: bool = False,
        merge_norm: str = "sqrt",
        learn_edge_gain: bool = True,
        learn_comp_gain: bool = True,
        edge_gain_scale: float = 0.5,
        distal_gain_init: float = 0.1,
        distal_residual_init: float = 0.0,
        distal_gate_alpha_init: float = 6.0,
        distal_gate_threshold_init: float = 0.5,
        input_residual_init: float = 0.0,
        detach_state_during_forward: bool = False,
        parallel_forward: bool = True,
        shared_tau_parallel: bool = True,
        integration_backend: str = "fft",
        branch_readout_mode: str = "trunk_distal",
        free_window_order: Optional[int] = None,
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        if channels is None:
            channels = c_sub
        if channels is None:
            raise ValueError("channels or c_sub must be provided")
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if compartments_per_branch < 2:
            raise ValueError("compartments_per_branch must be at least 2")
        if branch_degree <= 0:
            raise ValueError("branch_degree must be positive")
        if branch_assignment not in {"window", "cyclic"}:
            raise ValueError("branch_assignment must be 'window' or 'cyclic'")
        if merge_norm not in {"sqrt", "mean", "sum"}:
            raise ValueError("merge_norm must be one of: 'sqrt', 'mean', 'sum'")
        if branch_readout_mode not in {"trunk_distal", "linear"}:
            raise ValueError("branch_readout_mode must be 'trunk_distal' or 'linear'")
        if integration_backend not in {"gemm", "fft"}:
            raise ValueError("integration_backend must be 'gemm' or 'fft'")
        if tau_min <= 1.0 or tau_max <= 1.0:
            raise ValueError("tau_min and tau_max should be larger than 1")
        if free_window_order is not None:
            if isinstance(free_window_order, bool) or not isinstance(free_window_order, int):
                raise TypeError("free_window_order must be None or a positive integer")
            if free_window_order <= 0:
                raise ValueError("free_window_order must be positive when enabled")
            if no_filter:
                raise ValueError("free_window_order and no_filter cannot be enabled together")
            if step_mode != "m":
                raise ValueError("free_window_order requires step_mode='m'")
            if not parallel_forward:
                raise ValueError("free_window_order requires parallel_forward=True")
            if detach_state_during_forward:
                raise ValueError("free_window_order requires detach_state_during_forward=False")

        self.channels = int(channels)
        self.c_sub = self.channels
        self.num_branches = int(num_branches)
        self.compartments_per_branch = int(compartments_per_branch)
        self.distal_count = self.compartments_per_branch - 1
        self.branch_degree = min(int(branch_degree), self.num_branches)
        self.branch_assignment = branch_assignment
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.no_filter = no_filter
        self.merge_norm = merge_norm
        self.learn_edge_gain = bool(learn_edge_gain)
        self.learn_comp_gain = bool(learn_comp_gain)
        self.edge_gain_scale = float(edge_gain_scale)
        self.detach_state_during_forward = bool(detach_state_during_forward)
        self.parallel_forward = bool(parallel_forward)
        self.shared_tau_parallel = bool(shared_tau_parallel)
        self.integration_backend = integration_backend
        self.free_window_order = free_window_order
        self.branch_readout_mode = branch_readout_mode
        self.store_branch_monitor = bool(store_branch_monitor)

        active_branch_index = self._make_active_branch_index(
            self.channels,
            self.num_branches,
            self.branch_degree,
            self.branch_assignment,
        )
        channel_degree = torch.full((self.channels,), float(self.branch_degree))
        branch_usage = torch.bincount(
            active_branch_index.reshape(-1),
            minlength=self.num_branches,
        ).float()
        self.register_buffer("active_branch_index", active_branch_index)
        self.register_buffer("channel_degree", channel_degree)
        self.register_buffer("branch_usage", branch_usage)

        tau_data = self._make_tau_data(
            init_tau=init_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            compartment_tau_scale=compartment_tau_scale,
        )
        self.tau_compartments = nn.Parameter(
            tau_data.clone().detach().float(),
            requires_grad=self.free_window_order is None,
        )
        if self.free_window_order is None:
            self.register_parameter("free_window_weight", None)
        else:
            free_window_weight = torch.ones(self.num_branches, self.free_window_order)
            nn.init.kaiming_uniform_(free_window_weight, a=math.sqrt(5))  # sqrt(5) default for uniform init for window
            self.free_window_weight = nn.Parameter(free_window_weight)

        if self.learn_edge_gain:
            self.edge_channel_gain = nn.Parameter(torch.zeros(self.num_branches, self.channels))
        else:
            self.register_buffer("edge_channel_gain", torch.zeros(self.num_branches, self.channels))

        self.compartment_input_logits = nn.Parameter(torch.zeros(self.num_branches, self.compartments_per_branch))
        self.distal_mix_logits = nn.Parameter(torch.zeros(self.num_branches, self.distal_count))
        self.distal_gate_alpha = nn.Parameter(torch.full((self.num_branches, self.distal_count), float(distal_gate_alpha_init)))
        self.distal_gate_threshold = nn.Parameter(torch.full((self.num_branches, self.distal_count), float(distal_gate_threshold_init)))
        self.distal_gain = nn.Parameter(torch.full((self.num_branches,), float(distal_gain_init)))
        self.distal_residual_gain = nn.Parameter(torch.full((self.num_branches,), float(distal_residual_init)))
        self.branch_strength = nn.Parameter(torch.ones(self.num_branches))
        # Physical-space readout: C starts at the terminal compartment, while
        # the branch-level direct feedthrough D starts disabled.
        linear_comp_readout = torch.zeros(self.num_branches, self.compartments_per_branch)
        linear_comp_readout[:, -1] = 1.0
        self.linear_comp_readout = nn.Parameter(linear_comp_readout)
        self.linear_input_readout = nn.Parameter(torch.zeros(self.num_branches))
        self.input_residual_scale = nn.Parameter(torch.tensor(float(input_residual_init)))
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self.branch_input_seq = None
        self.branch_mod_seq = None
        self.distal_gate_seq = None
        self.branch_output_seq = None

    @staticmethod
    def _make_active_branch_index(
        channels: int,
        num_branches: int,
        branch_degree: int,
        branch_assignment: str,
    ):
        channel_index = torch.arange(channels, dtype=torch.long)
        if branch_assignment == "window":
            base = torch.div(channel_index * num_branches, channels, rounding_mode="floor")
        else:
            base = channel_index % num_branches

        edges = []
        for offset in range(branch_degree):
            edges.append((base + offset) % num_branches)
        return torch.stack(edges, dim=0)

    def _make_tau_data(self, init_tau, tau_min, tau_max, compartment_tau_scale):
        if init_tau is not None:
            tau_data = torch.as_tensor(init_tau, dtype=torch.float32)
            if tau_data.numel() == 1:
                return torch.full((self.num_branches, self.compartments_per_branch), float(tau_data.flatten()[0]))
            if tau_data.numel() == 2:
                tau_min_v = float(tau_data.flatten()[0])
                tau_max_v = float(tau_data.flatten()[-1])
                branch_tau = torch.exp(torch.linspace(
                    torch.log(torch.tensor(tau_min_v)),
                    torch.log(torch.tensor(tau_max_v)),
                    steps=self.num_branches,
                ))
                comp_scale = torch.linspace(1.0, 1.5, steps=self.compartments_per_branch)
                return branch_tau[:, None] * comp_scale[None, :]
            if tau_data.numel() == self.num_branches:
                return tau_data.flatten()[:, None].repeat(1, self.compartments_per_branch)
            if tau_data.numel() == self.compartments_per_branch:
                return tau_data.flatten()[None, :].repeat(self.num_branches, 1)
            if tau_data.numel() == self.num_branches * self.compartments_per_branch:
                return tau_data.reshape(self.num_branches, self.compartments_per_branch)
            raise ValueError(
                "init_tau must be scalar, length 2, length num_branches, "
                "length compartments_per_branch, or length num_branches * compartments_per_branch"
            )

        branch_tau = torch.exp(torch.linspace(
            torch.log(torch.tensor(float(tau_min))),
            torch.log(torch.tensor(float(tau_max))),
            steps=self.num_branches,
        ))
        if isinstance(compartment_tau_scale, (float, int)):
            comp_scale = torch.full((self.compartments_per_branch,), float(compartment_tau_scale))
        else:
            comp_scale_data = torch.as_tensor(compartment_tau_scale, dtype=torch.float32)
            if comp_scale_data.numel() == 2:
                comp_scale = torch.linspace(
                    float(comp_scale_data.flatten()[0]),
                    float(comp_scale_data.flatten()[-1]),
                    steps=self.compartments_per_branch,
                )
            elif comp_scale_data.numel() == self.compartments_per_branch:
                comp_scale = comp_scale_data.flatten()
            else:
                raise ValueError(
                    "compartment_tau_scale must be scalar, length 2, "
                    "or length compartments_per_branch"
                )
        return branch_tau[:, None] * comp_scale[None, :]

    def _assert_input_shape(self, x: torch.Tensor):
        if x.dim() < 2:
            raise ValueError("Expected input shape [N, C, ...]")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, but got {x.shape[1]}")

    def _edge_index(self, device=None):
        if device is None:
            return self.active_branch_index
        return self.active_branch_index.to(device=device)

    def _edge_branch_value(self, value: torch.Tensor):
        return value[self._edge_index(value.device)]

    def _edge_channel_value(self, value: torch.Tensor):
        edge_index = self._edge_index(value.device)
        channel_index = torch.arange(self.channels, device=value.device).view(1, self.channels)
        channel_index = channel_index.expand_as(edge_index)
        return value[edge_index, channel_index]

    def _view_edge_channel(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.branch_degree, self.channels, *([1] * (ref.dim() - 3)))

    def _view_edge_compartment(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.branch_degree, self.channels, self.compartments_per_branch, *([1] * (ref.dim() - 4)))

    def _view_edge_distal(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.branch_degree, self.channels, self.distal_count, *([1] * (ref.dim() - 4)))

    def _view_channel_degree(self, ref: torch.Tensor):
        return self.channel_degree.to(dtype=ref.dtype, device=ref.device).view(
            1, self.channels, *([1] * (ref.dim() - 2))
        )

    def _edge_channel_weight(self, dtype, device):
        gain = self.edge_channel_gain.to(dtype=dtype, device=device)
        gain = self._edge_channel_value(gain)
        if self.learn_edge_gain:
            return 1.0 + self.edge_gain_scale * torch.tanh(gain)
        return torch.ones_like(gain)

    def _edge_compartment_input_gain(self, dtype, device):
        if self.learn_comp_gain == False:
            return torch.ones(
                (self.branch_degree, self.channels, self.compartments_per_branch),
                dtype=dtype,
                device=device,
            )
        raw = F.softplus(self.compartment_input_logits.to(dtype=dtype, device=device)) + 1e-4
        normalized = self.compartments_per_branch * raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return self._edge_branch_value(normalized)

    def _build_branch_input(self, x: torch.Tensor):
        self._assert_input_shape(x)
        x_edge = x.unsqueeze(1).unsqueeze(3)
        edge_gain = self._view_edge_channel(self._edge_channel_weight(x.dtype, x.device), x.unsqueeze(1))
        comp_gain = self._view_edge_compartment(self._edge_compartment_input_gain(x.dtype, x.device), x_edge)
        branch_input = x_edge * edge_gain.unsqueeze(3) * comp_gain
        if self.store_branch_monitor:
            self._branch_input_step = branch_input.detach()
        else:
            self._branch_input_step = None
        return branch_input

    def _init_state(self, branch_input: torch.Tensor):
        if isinstance(self.v, float):
            return torch.full_like(branch_input, self.v)
        if self.v.shape != branch_input.shape:
            return torch.full_like(branch_input, self.v_rest)
        return self.v

    def _integrate(self, branch_input: torch.Tensor, v_prev: torch.Tensor):
        if self.free_window_order is not None:
            raise RuntimeError(
                "free_window_order uses the original SPSN whole-sequence GEMM and "
                "is unavailable in single-step or sequential execution"
            )
        if self.no_filter:
            return branch_input

        tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-3)
        tau = self._edge_branch_value(tau.to(dtype=branch_input.dtype, device=branch_input.device))
        tau = self._view_edge_compartment(tau, branch_input)
        if self.decay_input:
            return v_prev + (branch_input - (v_prev - self.v_rest)) / tau
        return v_prev - (v_prev - self.v_rest) / tau + branch_input

    def _build_branch_input_sequence(self, x_seq: torch.Tensor):
        x_edge = x_seq.unsqueeze(2).unsqueeze(4)
        spatial_ones = [1] * (x_seq.dim() - 3)
        edge_gain = self._edge_channel_weight(x_seq.dtype, x_seq.device).view(
            1, 1, self.branch_degree, self.channels, 1, *spatial_ones
        )
        comp_gain = self._edge_compartment_input_gain(x_seq.dtype, x_seq.device).view(
            1, 1, self.branch_degree, self.channels, self.compartments_per_branch, *spatial_ones
        )
        return x_edge * edge_gain * comp_gain

    def _build_tau_terms(self, tau: torch.Tensor, T: int, dtype, device):
        tau = torch.clamp(tau.reshape(-1).to(dtype=dtype, device=device), min=1.0 + 1e-3)
        decay = 1.0 - 1.0 / tau
        t = torch.arange(T, dtype=dtype, device=device)
        diff = t[:, None] - t[None, :]
        mask = diff >= 0
        powers = decay[:, None, None] ** diff.clamp_min(0).unsqueeze(0)
        tau_matrix = torch.where(
            mask.unsqueeze(0),
            powers,
            torch.zeros((), dtype=dtype, device=device),
        )
        tau_vec_init = decay[:, None] ** (t + 1.0)
        tau_vec_rest = 1.0 - tau_vec_init
        return tau_matrix, tau_vec_init, tau_vec_rest, tau

    def _build_exponential_fft_terms(self, tau: torch.Tensor, T: int, dtype, device):
        work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        tau = torch.clamp(
            tau.to(dtype=work_dtype, device=device),
            min=1.0 + 1e-3,
        )
        decay = 1.0 - 1.0 / tau
        lags = torch.arange(T, dtype=work_dtype, device=device)
        powers = decay.unsqueeze(-1).pow(lags)
        kernel = powers / tau.unsqueeze(-1) if self.decay_input else powers
        init_factor = decay.unsqueeze(-1).pow(lags + 1.0)
        return kernel, init_factor

    def _gen_free_window_gemm_weight(self, T: int, dtype, device):
        """Build the original SPSN causal sliding matrix for every branch."""
        gemm_weight = torch.zeros(
            self.num_branches,
            T,
            T,
            dtype=dtype,
            device=device,
        )
        free_window_weight = self.free_window_weight.to(dtype=dtype, device=device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.free_window_order)
            length = min(end - start, self.free_window_order)
            gemm_weight[:, i, start:end] = free_window_weight[
                :, self.free_window_order - length:self.free_window_order
            ]
        return gemm_weight

    def _parallel_integrate_free_window(self, branch_input_seq: torch.Tensor):
        """Apply one branch-shared SPSN window to all its compartments."""
        T, N = branch_input_seq.shape[:2]
        spatial_shape = branch_input_seq.shape[5:]
        flat_size = self.branch_degree * self.channels * self.compartments_per_branch

        permute_to_edge_first = (2, 3, 4, 0, 1) + tuple(range(5, branch_input_seq.dim()))
        branch_input_flat = branch_input_seq.permute(permute_to_edge_first).reshape(flat_size, T, -1)
        gemm_weight = self._gen_free_window_gemm_weight(
            T,
            branch_input_seq.dtype,
            branch_input_seq.device,
        )

        state_flat = torch.empty_like(branch_input_flat)
        edge_branch = self._edge_index(branch_input_seq.device).reshape(-1)
        edge_ids = torch.arange(
            self.branch_degree * self.channels,
            device=branch_input_seq.device,
            dtype=torch.long,
        )
        comp_ids = torch.arange(
            self.compartments_per_branch,
            device=branch_input_seq.device,
            dtype=torch.long,
        )

        for branch_idx in range(self.num_branches):
            branch_edge_ids = edge_ids[edge_branch == branch_idx]
            if branch_edge_ids.numel() == 0:
                continue

            flat_ids = (
                branch_edge_ids[:, None] * self.compartments_per_branch
                + comp_ids[None, :]
            ).reshape(-1)
            drive = branch_input_flat[flat_ids]
            drive_2d = drive.permute(1, 0, 2).reshape(T, -1)
            state_2d = gemm_weight[branch_idx].matmul(drive_2d)
            state = state_2d.reshape(T, flat_ids.numel(), -1).permute(1, 0, 2)
            state_flat[flat_ids] = state.to(dtype=state_flat.dtype)

        edge_first_shape = (
            self.branch_degree,
            self.channels,
            self.compartments_per_branch,
            T,
            N,
            *spatial_shape,
        )
        state_edge_first = state_flat.reshape(edge_first_shape)
        permute_to_time_first = (3, 4, 0, 1, 2) + tuple(range(5, len(edge_first_shape)))
        return state_edge_first.permute(permute_to_time_first).contiguous()

    def _build_fft_drive(self, x_seq: torch.Tensor):
        """Keep the raw drive unexpanded so its FFT is shared by all edges."""
        spatial_dims = x_seq.dim() - 3
        spatial_ones = [1] * spatial_dims
        work_dtype = torch.float64 if x_seq.dtype == torch.float64 else torch.float32
        edge_gain = self._edge_channel_weight(work_dtype, x_seq.device).view(
            1,
            self.branch_degree,
            self.channels,
            1,
            *spatial_ones,
            1,
        )
        comp_gain = self._edge_compartment_input_gain(
            work_dtype,
            x_seq.device,
        ).view(
            1,
            self.branch_degree,
            self.channels,
            self.compartments_per_branch,
            *spatial_ones,
            1,
        )
        drive = x_seq.unsqueeze(2).unsqueeze(4)
        return drive, edge_gain * comp_gain

    def _parallel_integrate_free_window_fft(self, x_seq: torch.Tensor):
        """Apply branch-shared SPSN windows as causal FFT convolutions."""
        T = x_seq.shape[0]
        spatial_dims = x_seq.dim() - 3
        kernel_length = min(T, self.free_window_order)
        lag_kernel = self.free_window_weight.flip(-1)[..., :kernel_length]
        edge_kernel = self._edge_branch_value(lag_kernel)
        kernel = edge_kernel.view(
            1,
            self.branch_degree,
            self.channels,
            1,
            *([1] * spatial_dims),
            kernel_length,
        )
        drive, input_gain = self._build_fft_drive(x_seq)
        return _causal_fft_convolution_time_first(drive, kernel * input_gain)

    def _parallel_integrate_fft(
        self,
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
    ):
        """Apply all exponential compartment filters as FFT convolutions."""
        if self.no_filter:
            return self._build_branch_input_sequence(x_seq)

        T = x_seq.shape[0]
        spatial_dims = x_seq.dim() - 3
        kernel, init_factor = self._build_exponential_fft_terms(
            self.tau_compartments,
            T,
            x_seq.dtype,
            x_seq.device,
        )
        drive, input_gain = self._build_fft_drive(x_seq)
        state_seq = _causal_fft_template_convolution_time_first(
            drive,
            kernel,
            self._edge_index(kernel.device),
            input_gain,
        )

        init_factor = self._edge_branch_value(init_factor)
        init_factor = init_factor.movedim(-1, 0).view(
            T,
            1,
            self.branch_degree,
            self.channels,
            self.compartments_per_branch,
            *([1] * spatial_dims),
        )
        init_factor = init_factor.to(dtype=state_seq.dtype)
        state_seq = state_seq + self.v_rest
        state_seq = state_seq + init_factor * (v_init.unsqueeze(0) - self.v_rest)
        return state_seq

    def _parallel_integrate(self, branch_input_seq: torch.Tensor, v_init: torch.Tensor):
        if self.no_filter:
            return branch_input_seq

        T, N = branch_input_seq.shape[:2]
        spatial_shape = branch_input_seq.shape[5:]
        flat_size = self.branch_degree * self.channels * self.compartments_per_branch

        permute_to_edge_first = (2, 3, 4, 0, 1) + tuple(range(5, branch_input_seq.dim()))
        branch_input_flat = branch_input_seq.permute(permute_to_edge_first).reshape(flat_size, T, -1)

        v_permute_to_edge_first = (1, 2, 3, 0) + tuple(range(4, v_init.dim()))
        v_init_flat = v_init.permute(v_permute_to_edge_first).reshape(flat_size, -1)

        tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-3)
        tau = self._edge_branch_value(tau.to(dtype=branch_input_seq.dtype, device=branch_input_seq.device))
        tau_matrix, tau_vec_init, tau_vec_rest, tau_flat = self._build_tau_terms(
            tau,
            T,
            branch_input_seq.dtype,
            branch_input_seq.device,
        )

        if self.decay_input:
            drive = branch_input_flat / tau_flat[:, None, None]
        else:
            drive = branch_input_flat

        state_flat = torch.bmm(tau_matrix, drive)
        state_flat = state_flat + tau_vec_init[:, :, None] * v_init_flat[:, None, :]
        state_flat = state_flat + tau_vec_rest[:, :, None] * self.v_rest

        edge_first_shape = (
            self.branch_degree,
            self.channels,
            self.compartments_per_branch,
            T,
            N,
            *spatial_shape,
        )
        state_edge_first = state_flat.reshape(edge_first_shape)
        permute_to_time_first = (3, 4, 0, 1, 2) + tuple(range(5, len(edge_first_shape)))
        return state_edge_first.permute(permute_to_time_first).contiguous()

    def _parallel_integrate_shared_tau(self, branch_input_seq: torch.Tensor, v_init: torch.Tensor):
        if self.no_filter:
            return branch_input_seq

        # tau_compartments is already (num_branches, compartments); keep the
        # temporal kernel at that size instead of expanding it to every edge/channel.
        T, N = branch_input_seq.shape[:2]
        spatial_shape = branch_input_seq.shape[5:]
        flat_size = self.branch_degree * self.channels * self.compartments_per_branch

        permute_to_edge_first = (2, 3, 4, 0, 1) + tuple(range(5, branch_input_seq.dim()))
        branch_input_flat = branch_input_seq.permute(permute_to_edge_first).reshape(flat_size, T, -1)

        v_permute_to_edge_first = (1, 2, 3, 0) + tuple(range(4, v_init.dim()))
        v_init_flat = v_init.permute(v_permute_to_edge_first).reshape(flat_size, -1)

        tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-3)
        tau = tau.to(dtype=branch_input_seq.dtype, device=branch_input_seq.device)
        tau_matrix, tau_vec_init, tau_vec_rest, tau_flat = self._build_tau_terms(
            tau,
            T,
            branch_input_seq.dtype,
            branch_input_seq.device,
        )

        state_flat = torch.empty_like(branch_input_flat,dtype=branch_input_seq.dtype, device=branch_input_seq.device)
        edge_branch = self._edge_index(branch_input_seq.device).reshape(-1)
        edge_ids = torch.arange(
            self.branch_degree * self.channels,
            device=branch_input_seq.device,
            dtype=torch.long,
        )

        for branch_idx in range(self.num_branches):
            branch_edge_ids = edge_ids[edge_branch == branch_idx]
            if branch_edge_ids.numel() == 0:
                continue

            for comp_idx in range(self.compartments_per_branch):
                shared_idx = branch_idx * self.compartments_per_branch + comp_idx
                flat_ids = branch_edge_ids * self.compartments_per_branch + comp_idx
                drive = branch_input_flat[flat_ids]
                if self.decay_input:
                    drive = drive / tau_flat[shared_idx]

                drive_2d = drive.permute(1, 0, 2).reshape(T, -1)
                state_2d = tau_matrix[shared_idx].matmul(drive_2d)
                state = state_2d.reshape(T, flat_ids.numel(), -1).permute(1, 0, 2)
                state = state + tau_vec_init[shared_idx].view(1, T, 1) * v_init_flat[flat_ids].unsqueeze(1)
                state = state + tau_vec_rest[shared_idx].view(1, T, 1) * self.v_rest
                state = state.to(dtype=state_flat.dtype)
                state_flat[flat_ids] = state

        edge_first_shape = (
            self.branch_degree,
            self.channels,
            self.compartments_per_branch,
            T,
            N,
            *spatial_shape,
        )
        state_edge_first = state_flat.reshape(edge_first_shape)
        permute_to_time_first = (3, 4, 0, 1, 2) + tuple(range(5, len(edge_first_shape)))
        return state_edge_first.permute(permute_to_time_first).contiguous()

    def _normalize_distal_delta(self, delta: torch.Tensor):
        reduce_dims = [2] + list(range(4, delta.dim()))
        rms = torch.sqrt(delta.square().mean(dim=reduce_dims, keepdim=True) + 1e-6)
        return delta / rms

    def _compute_linear_branch_output(self, state: torch.Tensor, raw_input: torch.Tensor):
        """Apply the routed branch readout C_r z + D_r x to every active edge."""
        if raw_input is None:
            raise ValueError("raw_input is required when branch_readout_mode='linear'")

        comp_readout = self._edge_branch_value(
            self.linear_comp_readout.to(dtype=state.dtype, device=state.device)
        )
        input_readout = self._edge_branch_value(
            self.linear_input_readout.to(dtype=state.dtype, device=state.device)
        )
        comp_readout = self._view_edge_compartment(comp_readout, state)
        raw_input = raw_input.unsqueeze(1)
        input_readout = self._view_edge_channel(input_readout, raw_input)

        y = (state * comp_readout).sum(dim=3) + raw_input * input_readout

        if self.store_branch_monitor:
            self._branch_mod_step = torch.zeros_like(y).detach()
            self._distal_gate_step = torch.zeros_like(state[:, :, :, 1:, ...]).detach()
            self._branch_output_step = y.detach()
        else:
            self._branch_mod_step = None
            self._distal_gate_step = None
            self._branch_output_step = None
        return y

    def _compute_branch_output(self, state: torch.Tensor, raw_input: torch.Tensor = None):
        if self.branch_readout_mode == "linear":
            return self._compute_linear_branch_output(state, raw_input)

        trunk = state[:, :, :, 0, ...]
        distal = state[:, :, :, 1:, ...]
        delta = distal - trunk.unsqueeze(3)
        #delta_norm = self._normalize_distal_delta(delta)

        alpha = F.softplus(self._edge_branch_value(self.distal_gate_alpha).to(dtype=state.dtype, device=state.device))
        threshold = self._edge_branch_value(self.distal_gate_threshold.abs()).to(dtype=state.dtype, device=state.device)
        alpha = self._view_edge_distal(alpha, delta)
        threshold = self._view_edge_distal(threshold, delta)
        distal_gate = torch.sigmoid(alpha * delta.abs() - threshold)

        mix = torch.softmax(self.distal_mix_logits, dim=-1)
        mix = self._edge_branch_value(mix.to(dtype=state.dtype, device=state.device))
        mix = self._view_edge_distal(mix, delta)
        branch_mod = (mix * distal_gate * torch.tanh(delta)).sum(dim=3)
        branch_mod = torch.tanh(branch_mod)

        distal_gain = self._edge_branch_value(self.distal_gain).to(dtype=state.dtype, device=state.device)
        #residual_gain = self._edge_branch_value(self.distal_residual_gain).to(dtype=state.dtype, device=state.device)
        branch_strength = self._edge_branch_value(self.branch_strength).to(dtype=state.dtype, device=state.device)
        distal_gain = self._view_edge_channel(distal_gain, trunk)
        #residual_gain = self._view_edge_channel(residual_gain, trunk)
        branch_strength = self._view_edge_channel(branch_strength, trunk)

        y = branch_strength * (trunk + distal_gain * trunk * branch_mod) #+ residual_gain * branch_mod)  #

        if self.store_branch_monitor:
            self._branch_mod_step = branch_mod.detach()
            self._distal_gate_step = distal_gate.detach()
            self._branch_output_step = y.detach()
        else:
            self._branch_mod_step = None
            self._distal_gate_step = None
            self._branch_output_step = None
        return y

    def _merge_edges(self, edge_output: torch.Tensor, raw_input: torch.Tensor):
        y = edge_output.sum(dim=1)
        degree = self._view_channel_degree(y)
        if self.merge_norm == "sqrt":
            y = y / torch.sqrt(degree)
        elif self.merge_norm == "mean":
            y = y / degree
        y = self.output_scale * y
        if self.branch_readout_mode != "linear":
            y = y + self.input_residual_scale * raw_input
        return y

    def _compute_branch_output_sequence(self, state_seq: torch.Tensor, raw_input_seq: torch.Tensor = None):
        T, N = state_seq.shape[:2]
        spatial_shape = state_seq.shape[5:]
        state_flat = state_seq.reshape(
            T * N,
            self.branch_degree,
            self.channels,
            self.compartments_per_branch,
            *spatial_shape,
        )
        raw_input_flat = None
        if raw_input_seq is not None:
            raw_input_flat = raw_input_seq.reshape(
                T * N,
                self.channels,
                *spatial_shape,
            )
        edge_output_flat = self._compute_branch_output(state_flat, raw_input_flat)
        edge_output_seq = edge_output_flat.reshape(
            T,
            N,
            self.branch_degree,
            self.channels,
            *spatial_shape,
        )
        if self.store_branch_monitor:
            self.branch_mod_seq = self._branch_mod_step.reshape(
                T,
                N,
                self.branch_degree,
                self.channels,
                *spatial_shape,
            )
            self.distal_gate_seq = self._distal_gate_step.reshape(
                T,
                N,
                self.branch_degree,
                self.channels,
                self.distal_count,
                *spatial_shape,
            )
            self.branch_output_seq = self._branch_output_step.reshape(
                T,
                N,
                self.branch_degree,
                self.channels,
                *spatial_shape,
            )
        return edge_output_seq

    def _merge_edges_sequence(self, edge_output_seq: torch.Tensor, raw_input_seq: torch.Tensor):
        T, N = raw_input_seq.shape[:2]
        spatial_shape = raw_input_seq.shape[3:]
        edge_output_flat = edge_output_seq.reshape(
            T * N,
            self.branch_degree,
            self.channels,
            *spatial_shape,
        )
        raw_input_flat = raw_input_seq.reshape(T * N, self.channels, *spatial_shape)
        y_flat = self._merge_edges(edge_output_flat, raw_input_flat)
        return y_flat.reshape(T, N, self.channels, *spatial_shape)

    def _step(self, x: torch.Tensor, v_prev: torch.Tensor):
        branch_input = self._build_branch_input(x)
        state = self._integrate(branch_input, v_prev)
        edge_output = self._compute_branch_output(state, x)
        y = self._merge_edges(edge_output, x)
        return y, state

    def single_step_forward(self, x: torch.Tensor):
        branch_input = self._build_branch_input(x)
        v_prev = self._init_state(branch_input)
        state = self._integrate(branch_input, v_prev)
        edge_output = self._compute_branch_output(state, x)
        y = self._merge_edges(edge_output, x)
        self.v = state.detach()
        if self.store_v_seq:
            self.v_seq = state.detach().unsqueeze(0)
        if self.store_branch_monitor:
            self.branch_input_seq = self._branch_input_step
            self.branch_mod_seq = self._branch_mod_step
            self.distal_gate_seq = self._distal_gate_step
            self.branch_output_seq = self._branch_output_step
        else:
            self.branch_input_seq = None
            self.branch_mod_seq = None
            self.distal_gate_seq = None
            self.branch_output_seq = None
        return y

    def _sequential_multi_step_forward(self, x_seq: torch.Tensor):
        if x_seq.dim() < 3:
            raise ValueError("Expected multi-step input shape [T, N, C, ...]")
        if x_seq.shape[2] != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, but got {x_seq.shape[2]}")

        y_seq = []
        v_seq = [] if self.store_v_seq else None
        if self.store_branch_monitor:
            branch_input_seq = []
            branch_mod_seq = []
            distal_gate_seq = []
            branch_output_seq = []

        first_branch_input = self._build_branch_input(x_seq[0])
        v = self._init_state(first_branch_input)
        for t in range(x_seq.shape[0]):
            if t == 0:
                state = self._integrate(first_branch_input, v)
                edge_output = self._compute_branch_output(state, x_seq[t])
                y = self._merge_edges(edge_output, x_seq[t])
            else:
                y, state = self._step(x_seq[t], v)
            y_seq.append(y)

            if self.store_branch_monitor:
                branch_input_seq.append(self._branch_input_step)
                branch_mod_seq.append(self._branch_mod_step)
                distal_gate_seq.append(self._distal_gate_step)
                branch_output_seq.append(self._branch_output_step)
            if self.store_v_seq:
                v_seq.append(state.detach())
            v = state.detach() if self.detach_state_during_forward else state

        self.v = v.detach()
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        if self.store_branch_monitor:
            self.branch_input_seq = torch.stack(branch_input_seq)
            self.branch_mod_seq = torch.stack(branch_mod_seq)
            self.distal_gate_seq = torch.stack(distal_gate_seq)
            self.branch_output_seq = torch.stack(branch_output_seq)
        else:
            self.branch_input_seq = None
            self.branch_mod_seq = None
            self.distal_gate_seq = None
            self.branch_output_seq = None
        return torch.stack(y_seq)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if x_seq.dim() < 3:
            raise ValueError("Expected multi-step input shape [T, N, C, ...]")
        if x_seq.shape[2] != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, but got {x_seq.shape[2]}")
        if self.free_window_order is not None and (
            (not self.parallel_forward) or self.detach_state_during_forward
        ):
            raise RuntimeError(
                "free_window_order requires parallel_forward=True and "
                "detach_state_during_forward=False"
            )
        if (not self.parallel_forward) or self.detach_state_during_forward:
            return self._sequential_multi_step_forward(x_seq)

        if self.integration_backend == "fft":
            branch_input_step = self._build_branch_input(x_seq[0])
            if self.free_window_order is not None:
                state_seq = self._parallel_integrate_free_window_fft(x_seq)
            else:
                v_init = self._init_state(branch_input_step)
                state_seq = self._parallel_integrate_fft(x_seq, v_init)

            needs_branch_input = self.store_branch_monitor
            if needs_branch_input:
                branch_input_seq = (
                    state_seq
                    if self.no_filter
                    else self._build_branch_input_sequence(x_seq)
                )
            else:
                branch_input_seq = None
        else:
            branch_input_seq = self._build_branch_input_sequence(x_seq)
            if self.free_window_order is not None:
                state_seq = self._parallel_integrate_free_window(branch_input_seq)
            else:
                v_init = self._init_state(branch_input_seq[0])
                if self.shared_tau_parallel:
                    state_seq = self._parallel_integrate_shared_tau(branch_input_seq, v_init)
                else:
                    state_seq = self._parallel_integrate(branch_input_seq, v_init)
        edge_output_seq = self._compute_branch_output_sequence(state_seq, x_seq)
        y_seq = self._merge_edges_sequence(edge_output_seq, x_seq)

        self.v = state_seq[-1].detach()
        if self.store_v_seq:
            self.v_seq = state_seq.detach()
        if self.store_branch_monitor:
            self.branch_input_seq = branch_input_seq.detach()
            self._branch_input_step = self.branch_input_seq[-1]
            self._branch_mod_step = self.branch_mod_seq[-1]
            self._distal_gate_step = self.distal_gate_seq[-1]
            self._branch_output_step = self.branch_output_seq[-1]
        else:
            self.branch_input_seq = None
            self.branch_mod_seq = None
            self.distal_gate_seq = None
            self.branch_output_seq = None
            self._branch_input_step = None
            self._branch_mod_step = None
            self._distal_gate_step = None
            self._branch_output_step = None
        return y_seq

class CoupledSparseChannelPreservingTrunkDistalDendCompartment(SparseChannelPreservingTrunkDistalDendCompartment):
    """Sparse channel-preserving dendrite with directed intra-branch coupling.

    This variant keeps the routing, active-edge input expansion, trunk/distal
    readout, and edge merge from ``SparseChannelPreservingTrunkDistalDendCompartment``.
    The only changed part is the compartment dynamics: the diagonal independent
    transition over ``K`` compartments is replaced by a lower-bidiagonal
    proximal-to-distal transition.

    For a single active edge and channel, centered compartment voltages follow

        h_0[t] = a_0 h_0[t - 1] + drive_0[t]
        h_k[t] = a_k * ((1 - lam_k) h_k[t - 1] + lam_k h_{k - 1}[t - 1])
                 + drive_k[t], k > 0

    where ``a_k = 1 - 1 / tau_k``.  The total historical gain of compartment
    ``k`` remains ``a_k``, so coupling redistributes the previous state between
    self-memory and proximal input instead of increasing the recurrent gain.
    """

    def __init__(
        self,
        *args,
        proximal_coupling_init: float = 0.1,
        proximal_coupling_max: float = 0.5,
        **kwargs,
    ):
        kwargs.setdefault("parallel_forward", True)
        super().__init__(*args, **kwargs)
        if proximal_coupling_max <= 0.0 or proximal_coupling_max > 1.0:
            raise ValueError("proximal_coupling_max must be in (0, 1]")
        if proximal_coupling_init < 0.0 or proximal_coupling_init > proximal_coupling_max:
            raise ValueError("proximal_coupling_init must be in [0, proximal_coupling_max]")

        self.proximal_coupling_max = float(proximal_coupling_max)
        eps = 1e-4
        init_ratio = float(proximal_coupling_init) / self.proximal_coupling_max
        init_ratio = min(max(init_ratio, eps), 1.0 - eps)
        init_logit = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.proximal_coupling_logits = nn.Parameter(
            torch.full((self.num_branches, self.distal_count), float(init_logit))
        )

    def _edge_proximal_coupling(self, dtype, device):
        coupling = self.proximal_coupling_max * torch.sigmoid(
            self.proximal_coupling_logits.to(dtype=dtype, device=device)
        )
        return self._edge_branch_value(coupling)

    def _build_decay_terms(self, decay: torch.Tensor, T: int, dtype, device):
        decay = decay.reshape(-1).to(dtype=dtype, device=device).clamp(0.0, 1.0 - 1e-4)
        t = torch.arange(T, dtype=dtype, device=device)
        diff = t[:, None] - t[None, :]
        mask = diff >= 0
        powers = decay[:, None, None] ** diff.clamp_min(0).unsqueeze(0)
        decay_matrix = torch.where(
            mask.unsqueeze(0),
            powers,
            torch.zeros((), dtype=dtype, device=device),
        )
        decay_vec_init = decay[:, None] ** (t + 1.0)
        return decay_matrix, decay_vec_init, decay

    def _parallel_filter_from_decay(
        self,
        drive_seq: torch.Tensor,
        h_init: torch.Tensor,
        self_decay: torch.Tensor,
    ):
        T, N = drive_seq.shape[:2]
        spatial_shape = drive_seq.shape[4:]
        flat_size = self.branch_degree * self.channels

        permute_to_edge_first = (2, 3, 0, 1) + tuple(range(4, drive_seq.dim()))
        drive_flat = drive_seq.permute(permute_to_edge_first).reshape(flat_size, T, -1)

        init_permute_to_edge_first = (1, 2, 0) + tuple(range(3, h_init.dim()))
        h_init_flat = h_init.permute(init_permute_to_edge_first).reshape(flat_size, -1)

        decay_matrix, decay_vec_init, _ = self._build_decay_terms(
            self_decay,
            T,
            drive_seq.dtype,
            drive_seq.device,
        )
        state_flat = torch.bmm(decay_matrix, drive_flat)
        state_flat = state_flat + decay_vec_init[:, :, None] * h_init_flat[:, None, :]

        edge_first_shape = (
            self.branch_degree,
            self.channels,
            T,
            N,
            *spatial_shape,
        )
        state_edge_first = state_flat.reshape(edge_first_shape)
        permute_to_time_first = (2, 3, 0, 1) + tuple(range(4, len(edge_first_shape)))
        return state_edge_first.permute(permute_to_time_first).contiguous()

    def _integrate(self, branch_input: torch.Tensor, v_prev: torch.Tensor):
        if self.no_filter:
            return branch_input

        tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-3)
        tau = self._edge_branch_value(tau.to(dtype=branch_input.dtype, device=branch_input.device))
        tau = self._view_edge_compartment(tau, branch_input)
        decay = 1.0 - 1.0 / tau
        drive = branch_input / tau if self.decay_input else branch_input

        h_prev = v_prev - self.v_rest
        first_state = decay[:, :, :, :1, ...] * h_prev[:, :, :, :1, ...] + drive[:, :, :, :1, ...]

        coupling = self._edge_proximal_coupling(branch_input.dtype, branch_input.device)
        coupling = self._view_edge_distal(coupling, h_prev[:, :, :, 1:, ...])
        distal_decay = decay[:, :, :, 1:, ...]
        distal_state = (
            distal_decay * (1.0 - coupling) * h_prev[:, :, :, 1:, ...]
            + distal_decay * coupling * h_prev[:, :, :, :-1, ...]
            + drive[:, :, :, 1:, ...]
        )
        return torch.cat([first_state, distal_state], dim=3) + self.v_rest

    def _parallel_integrate(self, branch_input_seq: torch.Tensor, v_init: torch.Tensor):
        if self.no_filter:
            return branch_input_seq

        tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-3)
        tau = self._edge_branch_value(
            tau.to(dtype=branch_input_seq.dtype, device=branch_input_seq.device)
        )
        decay = 1.0 - 1.0 / tau
        coupling = self._edge_proximal_coupling(branch_input_seq.dtype, branch_input_seq.device)

        h_init = v_init - self.v_rest
        h_states = []
        for k in range(self.compartments_per_branch):
            input_k = branch_input_seq[:, :, :, :, k, ...]
            tau_k = tau[:, :, k]
            drive_k = input_k / tau_k.view(1, 1, self.branch_degree, self.channels, *([1] * (input_k.dim() - 4)))
            if not self.decay_input:
                drive_k = input_k

            h_init_k = h_init[:, :, :, k, ...]
            if k == 0:
                self_decay = decay[:, :, k]
            else:
                lam = coupling[:, :, k - 1]
                self_decay = decay[:, :, k] * (1.0 - lam)
                lower_decay = decay[:, :, k] * lam
                lower_shift = torch.cat(
                    [h_init[:, :, :, k - 1, ...].unsqueeze(0), h_states[k - 1][:-1]],
                    dim=0,
                )
                drive_k = drive_k + lower_decay.view(
                    1,
                    1,
                    self.branch_degree,
                    self.channels,
                    *([1] * (input_k.dim() - 4)),
                ) * lower_shift

            h_states.append(self._parallel_filter_from_decay(drive_k, h_init_k, self_decay))

        return torch.stack(h_states, dim=4) + self.v_rest


