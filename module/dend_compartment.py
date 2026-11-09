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

BRANCH_READOUT_MODES = ("trunk_distal", "linear", "linear_act", "glu", "swiglu", "mlp")
READOUT_ACTIVATIONS = {
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "silu": F.silu,
}


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

    ``branch_readout_mode`` selects the branch readout R(z) on the K compartment
    states z of each routed branch (parameters shared per branch template):

    * ``"trunk_distal"`` (default): DMPR.
    * ``"linear"``:     R = w^T z + b
    * ``"linear_act"``: R = act(w^T z + b), act = ``readout_activation``
    * ``"glu"``:        R = (w1^T z + b1) * sigmoid(w2^T z + b2)
    * ``"swiglu"``:     R = (w1^T z + b1) * SiLU(w2^T z + b2)
    * ``"mlp"``:        R = w2^T sigmoid(W1 z + b1) + b2, hidden width
      ``readout_mlp_hidden``

    Every readout feeds the same somatic input I = output_scale * R
    + input_residual_scale * X (Eq. 11).
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
        readout_activation: str = "gelu",
        readout_mlp_hidden: int = 2,
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
        if branch_readout_mode not in BRANCH_READOUT_MODES:
            raise ValueError(f"branch_readout_mode must be one of {BRANCH_READOUT_MODES}")
        if readout_activation not in READOUT_ACTIVATIONS:
            raise ValueError(f"readout_activation must be one of {tuple(READOUT_ACTIVATIONS)}")
        if int(readout_mlp_hidden) <= 0:
            raise ValueError("readout_mlp_hidden must be positive")
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
        self.readout_activation = readout_activation
        self.readout_mlp_hidden = int(readout_mlp_hidden)
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
        if self.branch_readout_mode == "trunk_distal":
            self.distal_mix_logits = nn.Parameter(torch.zeros(self.num_branches, self.distal_count))
            self.distal_gate_alpha = nn.Parameter(torch.full((self.num_branches, self.distal_count), float(distal_gate_alpha_init)))
            self.distal_gate_threshold = nn.Parameter(torch.full((self.num_branches, self.distal_count), float(distal_gate_threshold_init)))
            self.distal_gain = nn.Parameter(torch.full((self.num_branches,), float(distal_gain_init)))
            self.distal_residual_gain = nn.Parameter(torch.full((self.num_branches,), float(distal_residual_init)))
            self.branch_strength = nn.Parameter(torch.ones(self.num_branches))
        else:
            self._init_readout_parameters()
        self.input_residual_scale = nn.Parameter(torch.tensor(float(input_residual_init)))
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self.branch_input_seq = None
        self.branch_mod_seq = None
        self.distal_gate_seq = None
        self.branch_output_seq = None

    def _init_readout_parameters(self):
        """Template-shared parameters of the W1 readouts (Table R1-1).

        Weights and biases use the default ``nn.Linear`` initialization
        U(-1/sqrt(fan_in), 1/sqrt(fan_in)) of each projection.
        """
        B, K, H = self.num_branches, self.compartments_per_branch, self.readout_mlp_hidden

        def uniform(*shape, fan_in):
            bound = 1.0 / math.sqrt(fan_in)
            return nn.Parameter(torch.empty(*shape).uniform_(-bound, bound))

        mode = self.branch_readout_mode
        if mode in ("linear", "linear_act"):
            self.readout_weight = uniform(B, K, fan_in=K)               # w
            self.readout_bias = uniform(B, fan_in=K)                    # b
        elif mode in ("glu", "swiglu"):
            # index 0: value projection (w1, b1); index 1: gate projection (w2, b2)
            self.readout_weight = uniform(B, 2, K, fan_in=K)
            self.readout_bias = uniform(B, 2, fan_in=K)
        elif mode == "mlp":
            self.readout_hidden_weight = uniform(B, H, K, fan_in=K)     # W1
            self.readout_hidden_bias = uniform(B, H, fan_in=K)          # b1
            self.readout_out_weight = uniform(B, H, fan_in=H)           # w2
            self.readout_out_bias = uniform(B, fan_in=H)                # b2
        else:
            raise ValueError(f"no readout parameters for mode {mode!r}")

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

    def _compute_readout_branch_output(self, state: torch.Tensor):
        """W1 readouts (Table R1-1) on the raw compartment states.

        ``state`` is ``[M, d, C, K, *spatial]``; the result is ``[M, d, C, *spatial]``.
        Unlike DMPR, no RMS normalization is applied to the states.
        """
        dtype, device = state.dtype, state.device
        d, C = self.branch_degree, self.channels
        spatial = [1] * (state.dim() - 4)

        def routed(value):
            # [B, *rest] -> [d, C, *rest], the template of each active edge
            return self._edge_branch_value(value.to(dtype=dtype, device=device))

        def project(weight, bias):
            # sum_k weight[k] * z_k + bias with weight [B, K] and bias [B]
            w = routed(weight).view(1, d, C, self.compartments_per_branch, *spatial)
            b = routed(bias).view(1, d, C, *spatial)
            return (state * w).sum(dim=3) + b

        mode = self.branch_readout_mode
        if mode in ("linear", "linear_act"):
            y = project(self.readout_weight, self.readout_bias)
            if mode == "linear_act":
                y = READOUT_ACTIVATIONS[self.readout_activation](y)
        elif mode in ("glu", "swiglu"):
            value = project(self.readout_weight[:, 0], self.readout_bias[:, 0])
            gate = project(self.readout_weight[:, 1], self.readout_bias[:, 1])
            gate = torch.sigmoid(gate) if mode == "glu" else F.silu(gate)
            y = value * gate
        elif mode == "mlp":
            H = self.readout_mlp_hidden
            w1 = routed(self.readout_hidden_weight).view(1, d, C, H, self.compartments_per_branch, *spatial)
            b1 = routed(self.readout_hidden_bias).view(1, d, C, H, *spatial)
            hidden = torch.sigmoid((state.unsqueeze(3) * w1).sum(dim=4) + b1)
            w2 = routed(self.readout_out_weight).view(1, d, C, H, *spatial)
            b2 = routed(self.readout_out_bias).view(1, d, C, *spatial)
            y = (hidden * w2).sum(dim=3) + b2
        else:
            raise ValueError(f"unknown branch_readout_mode {mode!r}")

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
        if self.branch_readout_mode != "trunk_distal":
            return self._compute_readout_branch_output(state)

        trunk = state[:, :, :, 0, ...]
        distal = state[:, :, :, 1:, ...]
        delta = distal - trunk.unsqueeze(3)
        delta_norm = self._normalize_distal_delta(delta)

        alpha = F.softplus(self._edge_branch_value(self.distal_gate_alpha).to(dtype=state.dtype, device=state.device))
        threshold = self._edge_branch_value(self.distal_gate_threshold.abs()).to(dtype=state.dtype, device=state.device)
        alpha = self._view_edge_distal(alpha, delta_norm)
        threshold = self._view_edge_distal(threshold, delta_norm)
        distal_gate = torch.sigmoid(alpha * delta_norm.abs() - threshold)

        mix = torch.softmax(self.distal_mix_logits, dim=-1)
        mix = self._edge_branch_value(mix.to(dtype=state.dtype, device=state.device))
        mix = self._view_edge_distal(mix, delta_norm)
        branch_mod = (mix * distal_gate * torch.tanh(delta_norm)).sum(dim=3)
        branch_mod = torch.tanh(branch_mod)

        distal_gain = self._edge_branch_value(self.distal_gain).to(dtype=state.dtype, device=state.device)
        residual_gain = self._edge_branch_value(self.distal_residual_gain).to(dtype=state.dtype, device=state.device)
        branch_strength = self._edge_branch_value(self.branch_strength).to(dtype=state.dtype, device=state.device)
        distal_gain = self._view_edge_channel(distal_gain, trunk)
        residual_gain = self._view_edge_channel(residual_gain, trunk)
        branch_strength = self._view_edge_channel(branch_strength, trunk)

        y = branch_strength * (trunk + distal_gain * trunk * branch_mod + residual_gain * branch_mod)

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
        # Eq. 11: I = eta * R + xi * X, identical for every branch readout.
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



class ChainCoupledSparseDendCompartment(SparseChannelPreservingTrunkDistalDendCompartment):
    """Rebuttal W5 (Table R1-7): symmetric tridiagonal intra-branch coupling.

    Compartments 0 (proximal) -> K-1 form a chain. For branch template j the
    diagonal transition of Eq. 2 is replaced by

        A_j = I - diag(1 / tau_j) - L(c_j),
        L(c)_kk = c_{k-1,k} + c_{k,k+1},   L(c)_{k,k+1} = L(c)_{k+1,k} = -c_{k,k+1},

    i.e. a passive chain Laplacian (diffusion between neighbouring compartments).
    Each coupling is c_{k,k+1} = c_max * sigmoid(coupling_logits) with

        c_max = min(1 - 1/tau_k, 1 - 1/tau_{k+1}) / 4   (computed from the current
                                                         tau, gradient detached),

    so A_j is symmetric with all eigenvalues in (0, 1) (Gershgorin): real,
    stable and non-oscillatory, like the diagonal model. Couplings start at
    ``coupling_init_ratio * c_max``. The input vector is unchanged
    (b = g_c * q / tau), and every branch readout of the parent class works.

    Compartment states are the causal convolutions z_t = sum_l A^l b x_{t-l};
    the K-dimensional kernels A^l b are built by recursion and applied with
    the same template FFT as the diagonal model. Only the parallel FFT path is
    supported (integration_backend='fft', step_mode='m').
    """

    def __init__(self, *args, coupling_init_ratio: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 < coupling_init_ratio < 1.0:
            raise ValueError("coupling_init_ratio must be in (0, 1)")
        if (
            self.integration_backend != "fft"
            or not self.parallel_forward
            or self.detach_state_during_forward
            or self.free_window_order is not None
            or self.no_filter
        ):
            raise ValueError(
                "ChainCoupledSparseDendCompartment requires integration_backend='fft', "
                "parallel_forward=True, detach_state_during_forward=False, "
                "free_window_order=None and no_filter=False"
            )
        self.coupling_init_ratio = float(coupling_init_ratio)
        init_logit = math.log(coupling_init_ratio / (1.0 - coupling_init_ratio))
        self.coupling_logits = nn.Parameter(
            torch.full((self.num_branches, self.compartments_per_branch - 1), init_logit)
        )

    def coupling_strength(self, dtype=torch.float32, device=None):
        """Return (c [B, K-1], c_max [B, K-1]) for every branch template."""
        device = self.coupling_logits.device if device is None else device
        tau = torch.clamp(self.tau_compartments.to(dtype=dtype, device=device), min=1.0 + 1e-3)
        decay = (1.0 - 1.0 / tau).detach()
        c_max = torch.minimum(decay[:, :-1], decay[:, 1:]) / 4.0
        c = c_max * torch.sigmoid(self.coupling_logits.to(dtype=dtype, device=device))
        return c, c_max

    def transition_matrix(self, dtype=torch.float32, device=None):
        """A_j = I - diag(1/tau_j) - L(c_j), shape [B, K, K]."""
        device = self.coupling_logits.device if device is None else device
        K = self.compartments_per_branch
        tau = torch.clamp(self.tau_compartments.to(dtype=dtype, device=device), min=1.0 + 1e-3)
        c, _ = self.coupling_strength(dtype, device)
        zeros = torch.zeros_like(c[:, :1])
        degree = torch.cat([zeros, c], dim=1) + torch.cat([c, zeros], dim=1)   # c_{k-1,k} + c_{k,k+1}
        eye = torch.eye(K, dtype=dtype, device=device)
        A = eye * (1.0 - 1.0 / tau - degree).unsqueeze(-1)
        A = A + torch.diag_embed(c, offset=1) + torch.diag_embed(c, offset=-1)
        return A

    def _template_compartment_input_gain(self, dtype, device):
        """q_j of Eq. 4 per branch template, shape [B, K]."""
        if not self.learn_comp_gain:
            return torch.ones(self.num_branches, self.compartments_per_branch, dtype=dtype, device=device)
        raw = F.softplus(self.compartment_input_logits.to(dtype=dtype, device=device)) + 1e-4
        return self.compartments_per_branch * raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _integrate(self, branch_input: torch.Tensor, v_prev: torch.Tensor):
        raise RuntimeError(
            "ChainCoupledSparseDendCompartment supports only the parallel FFT "
            "multi-step path (step_mode='m')"
        )

    def _parallel_integrate_fft(self, x_seq: torch.Tensor, v_init: torch.Tensor):
        T = x_seq.shape[0]
        spatial_dims = x_seq.dim() - 3
        work_dtype = torch.float64 if x_seq.dtype == torch.float64 else torch.float32
        device = x_seq.device

        A = self.transition_matrix(work_dtype, device)                       # [B, K, K]
        tau = torch.clamp(self.tau_compartments.to(dtype=work_dtype, device=device), min=1.0 + 1e-3)
        q = self._template_compartment_input_gain(work_dtype, device)
        b = q / tau if self.decay_input else q                               # [B, K]

        # kernel[j, :, l] = A_j^l b_j  (q is mixed by A, so it lives in the kernel)
        columns = []
        v = b
        for _ in range(T):
            columns.append(v)
            v = torch.einsum("bkm,bm->bk", A, v)
        kernel = torch.stack(columns, dim=-1)                                # [B, K, T]

        drive = x_seq.unsqueeze(2).unsqueeze(4)
        edge_gain = self._edge_channel_weight(work_dtype, device).view(
            1, self.branch_degree, self.channels, 1, *([1] * spatial_dims), 1
        )
        state_seq = _causal_fft_template_convolution_time_first(
            drive, kernel, self._edge_index(device), edge_gain,
        )

        # A carried state (no reset between calls) propagates as A^{t+1} h0.
        zero_init = (
            (isinstance(self.v, float) and self.v == self.v_rest)
            or (torch.is_tensor(self.v) and self.v.shape != v_init.shape)
        )
        if not zero_init:
            A_edge = self._edge_branch_value(A).view(
                1, self.branch_degree, self.channels,
                self.compartments_per_branch, self.compartments_per_branch,
                *([1] * spatial_dims),
            ).to(dtype=state_seq.dtype)
            h = (v_init - self.v_rest).to(dtype=state_seq.dtype)
            init_terms = []
            for _ in range(T):
                h = (A_edge * h.unsqueeze(3)).sum(dim=4)
                init_terms.append(h)
            state_seq = state_seq + torch.stack(init_terms, dim=0)
        return state_seq + self.v_rest


