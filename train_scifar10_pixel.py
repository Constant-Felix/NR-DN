"""Train pixel-by-pixel CIFAR-10 with the paper backbone and DEND+SOMA neurons.

The data layout, surrounding network, and optimizer defaults follow the
official length-1024 Sequential CIFAR-10 implementation released with:

    PMSN: A Parallel Multi-compartment Spiking Neuron for Multi-scale
    Temporal Processing (arXiv:2408.14917v2)

Every PMSN temporal layer is replaced by the following local modules:

    SparseChannelPreservingTrunkDistalDendCompartment
        -> MaskedSlidingPSN or PSNIntergerSoma_ssf

CIFAR-10 images are normalized and scanned in row-major pixel order.  Each
time step is one RGB vector, so the model input has shape [B, 1024, 3].
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_SEQUENCE_LENGTH = 32 * 32
LOGGER_NAME = "pixel_cifar10_dend_soma"


def _load_local_class(relative_path: str, module_name: str, class_name: str):
    """Load a project class without executing ``module/__init__.py``.

    The repository still contains legacy ``spikingjelly.clock_driven`` imports
    in package initializers.  Loading the two modern activation-based files
    directly keeps this program compatible with the installed SpikingJelly.
    """

    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        path = Path(__file__).resolve().parent / relative_path
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {class_name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return getattr(module, class_name)


def image_to_pixel_sequence(image: Tensor) -> Tensor:
    """Convert ``[C, H, W]`` to row-major ``[H * W, C]``."""

    if image.ndim != 3:
        raise ValueError(f"Expected [C, H, W], got {tuple(image.shape)}")
    channels, height, width = image.shape
    return image.reshape(channels, height * width).transpose(0, 1).contiguous()


class PixelSequence:
    """Pickle-friendly torchvision transform for pixel-by-pixel scanning."""

    def __call__(self, image: Tensor) -> Tensor:
        return image_to_pixel_sequence(image)


class TiedDropout(nn.Module):
    """Drop complete feature channels while sharing the mask over time."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.probability = float(probability)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return x
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")
        keep_probability = 1.0 - self.probability
        mask = torch.rand(
            x.shape[0], x.shape[1], 1, device=x.device
        ) < keep_probability
        return x * mask.to(dtype=x.dtype) / keep_probability


@dataclass(frozen=True)
class DendSomaConfig:
    soma_type: str = "masked_sliding_psn"
    psn_order: int = CIFAR10_SEQUENCE_LENGTH
    psn_backend: str = "fft"
    psn_exp_init: bool = False
    integer_psn_bias: float = 0.0
    ssf_thre: int = 4
    num_branches: int = 4
    compartments_per_branch: int = 2
    branch_degree: int = 1
    branch_assignment: str = "cyclic"
    dend_backend: str = "fft"
    dend_merge_norm: str = "sqrt"
    learn_edge_gain: bool = False
    learn_comp_gain: bool = True
    activation_checkpoint: bool = True

    def __post_init__(self) -> None:
        supported_somas = {"masked_sliding_psn", "psn_integer_ssf"}
        if self.soma_type not in supported_somas:
            raise ValueError(
                f"soma_type must be one of {sorted(supported_somas)}, "
                f"got {self.soma_type!r}"
            )
        if self.psn_order <= 0:
            raise ValueError("psn_order must be positive")
        if self.psn_backend not in {"gemm", "conv", "fft"}:
            raise ValueError("psn_backend must be gemm, conv, or fft")
        if (
            self.soma_type == "masked_sliding_psn"
            and self.psn_backend == "conv"
            and self.psn_exp_init
        ):
            raise ValueError(
                "MaskedSlidingPSN backend='conv' ignores exp_init; use "
                "backend='gemm'/'fft' or disable psn_exp_init"
            )
        if self.dend_backend not in {"gemm", "fft"}:
            raise ValueError("dend_backend must be gemm or fft")
        if self.num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if self.compartments_per_branch < 2:
            raise ValueError("compartments_per_branch must be at least 2")
        if self.branch_degree <= 0:
            raise ValueError("branch_degree must be positive")
        if self.ssf_thre <= 0:
            raise ValueError("ssf_thre must be positive")


@dataclass(frozen=True)
class ModelConfig:
    d_input: int = 3
    d_output: int = 10
    d_model: int = 128
    n_layers: int = 3
    dropout: float = 0.1
    prenorm: bool = True
    norm: str = "BN"

    def __post_init__(self) -> None:
        if self.d_input <= 0 or self.d_output <= 0:
            raise ValueError("d_input and d_output must be positive")
        if self.d_model < 4:
            raise ValueError("d_model must be at least 4")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.norm not in {"BN", "LN", "None"}:
            raise ValueError("norm must be BN, LN, or None")


class DendSomaTemporalLayer(nn.Module):
    """PMSN-compatible temporal layer backed by local DEND+SOMA modules.

    External tensors use the paper implementation's ``[B, C, T]`` layout.
    The local dendrite and soma both consume ``[T, B, C]``.  The dendritic
    state is reset inside the checkpointed function so independent CIFAR-10
    batches cannot leak state and backward recomputation starts from zero.
    """

    def __init__(
        self,
        d_model: int,
        config: DendSomaConfig,
        neuron_dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.config = config

        dend_class = _load_local_class(
            "module/dend_compartment.py",
            "_pixel_cifar_dend_compartment",
            "SparseChannelPreservingTrunkDistalDendCompartment",
        )
        self.dend = dend_class(
            channels=self.d_model,
            num_branches=config.num_branches,
            compartments_per_branch=config.compartments_per_branch,
            branch_degree=config.branch_degree,
            branch_assignment=config.branch_assignment,
            step_mode="m",
            store_v_seq=False,
            store_branch_monitor=False,
            merge_norm=config.dend_merge_norm,
            learn_edge_gain=config.learn_edge_gain,
            learn_comp_gain=config.learn_comp_gain,
            detach_state_during_forward=False,
            parallel_forward=True,
            integration_backend=config.dend_backend,
            free_window_order=None,
        )

        if config.soma_type == "masked_sliding_psn":
            soma_class = _load_local_class(
                "module/soma.py",
                "_pixel_cifar_soma",
                "MaskedSlidingPSN",
            )
            self.soma = soma_class(
                order=config.psn_order,
                exp_init=config.psn_exp_init,
                backend=config.psn_backend,
            )
        else:
            soma_class = _load_local_class(
                "module/soma.py",
                "_pixel_cifar_soma",
                "PSNIntergerSoma_ssf",
            )
            self.soma = soma_class(
                psn_order=config.psn_order,
                psn_exp_init=config.psn_exp_init,
                psn_backend=config.psn_backend,
                psn_threshold_init=config.integer_psn_bias,
                thre=config.ssf_thre,
                step_mode="m",
            )

        # The original PMSN layer applies a tied dropout of dropout / 5.
        self.dropout = TiedDropout(neuron_dropout)

    def reset_state(self) -> None:
        if hasattr(self.dend, "reset"):
            self.dend.reset()
        if hasattr(self.soma, "reset"):
            self.soma.reset()
        if hasattr(self.soma, "firing_rate"):
            self.soma.firing_rate = 0.0

    def _forward_time_first(self, x_seq: Tensor) -> Tensor:
        self.reset_state()
        output = self.soma(self.dend(x_seq))
        if hasattr(self.soma, "firing_rate"):
            self.soma.firing_rate = output.detach().float().mean()
        return output

    def _apply_dend_soma(self, x_seq: Tensor) -> Tensor:
        if (
            self.config.activation_checkpoint
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                self._forward_time_first,
                x_seq,
                use_reentrant=False,
            )
        return self._forward_time_first(x_seq)

    def forward(
        self,
        u: Tensor,
        residual_spike: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if u.ndim != 3 or u.shape[1] != self.d_model:
            raise ValueError(
                f"Expected [B, {self.d_model}, T], got {tuple(u.shape)}"
            )
        x_seq = u.permute(2, 0, 1).contiguous()
        spike = self._apply_dend_soma(x_seq).permute(1, 2, 0).contiguous()
        if residual_spike is not None:
            if residual_spike.shape != spike.shape:
                raise ValueError(
                    "Residual spike shape mismatch: "
                    f"{tuple(residual_spike.shape)} != {tuple(spike.shape)}"
                )
            spike = spike + residual_spike
        spike = self.dropout(spike)
        return spike, spike


def _make_norm(norm: str, d_model: int) -> nn.Module:
    if norm == "BN":
        return nn.BatchNorm1d(d_model)
    if norm == "LN":
        return nn.LayerNorm(d_model)
    if norm == "None":
        return nn.Identity()
    raise ValueError(f"Unknown norm {norm!r}")


class PixelCIFAR10DendSomaModel(nn.Module):
    """Official pixel-CIFAR backbone with all six PMSNs replaced."""

    def __init__(
        self,
        model_config: ModelConfig,
        dend_soma_config: DendSomaConfig,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.dend_soma_config = dend_soma_config
        d_model = model_config.d_model
        dropout = model_config.dropout

        self.encoder = nn.Linear(model_config.d_input, d_model)
        (
            self.temporal_layers1,
            self.linear_layers1,
            self.norms1,
            self.dropouts1,
        ) = self._make_stage(d_model)

        # These operations deliberately pool the hidden dimension, matching
        # the official seqcifar1024 implementation rather than pooling time.
        self.avgpool1 = nn.AvgPool1d(4)
        self.fc1 = nn.Linear(d_model // 4, d_model * 2)

        d_model2 = d_model * 2
        (
            self.temporal_layers2,
            self.linear_layers2,
            self.norms2,
            self.dropouts2,
        ) = self._make_stage(d_model2)

        self.avgpool2 = nn.AvgPool1d(4)
        self.decoder = nn.Linear(d_model // 2, model_config.d_output)

    def _make_stage(self, d_model: int):
        temporal_layers = nn.ModuleList()
        linear_layers = nn.ModuleList()
        norms = nn.ModuleList()
        dropouts = nn.ModuleList()
        for _ in range(self.model_config.n_layers):
            temporal_layers.append(
                DendSomaTemporalLayer(
                    d_model=d_model,
                    config=self.dend_soma_config,
                    neuron_dropout=self.model_config.dropout / 5.0,  ##
                    #neuron_dropout=0
                )
            )
            linear_layers.append(nn.Conv1d(d_model, d_model, kernel_size=1))
            norms.append(_make_norm(self.model_config.norm, d_model))
            dropouts.append(nn.Dropout1d(self.model_config.dropout))
        return temporal_layers, linear_layers, norms, dropouts

    def _apply_stage(
        self,
        x: Tensor,
        temporal_layers: nn.ModuleList,
        linear_layers: nn.ModuleList,
        norms: nn.ModuleList,
        dropouts: nn.ModuleList,
    ) -> Tensor:
        residual_spike: Optional[Tensor] = None
        for temporal, linear, norm, dropout in zip(
            temporal_layers, linear_layers, norms, dropouts
        ):
            z = x
            if self.model_config.prenorm:
                if self.model_config.norm in {"BN", "None"}:
                    z = norm(z)
                else:
                    z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, residual_spike = temporal(z, residual_spike)
            x = dropout(linear(z))

        if residual_spike is None:
            raise RuntimeError("A stage must contain at least one temporal layer")
        return dropouts[-1](residual_spike).transpose(-1, -2)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.model_config.d_input:
            raise ValueError(
                f"Expected [B, T, {self.model_config.d_input}], got {tuple(x.shape)}"
            )
        x = self.encoder(x).transpose(-1, -2)

        x = self._apply_stage(
            x,
            self.temporal_layers1,
            self.linear_layers1,
            self.norms1,
            self.dropouts1,
        )
        x = self.fc1(self.avgpool1(x)).transpose(-1, -2)

        x = self._apply_stage(
            x,
            self.temporal_layers2,
            self.linear_layers2,
            self.norms2,
            self.dropouts2,
        )
        x = self.avgpool2(x).mean(dim=1)
        return self.decoder(x)


@dataclass
class EpochMetrics:
    loss: float
    top1: float
    top5: float
    samples: int
    seconds: float


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(
            [device_state.cpu() for device_state in state["cuda"]]
        )


def build_cifar10_transform() -> transforms.Compose:
    """The exact no-augmentation transform used by the official code."""

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            PixelSequence(),
        ]
    )


def build_dataloaders(
    data_path: str,
    batch_size: int,
    num_workers: int,
    download: bool,
    seed: int,
    pin_memory: bool,
    generator_state: Optional[Tensor] = None,
) -> Tuple[DataLoader, DataLoader, torch.Generator]:
    transform = build_cifar10_transform()
    train_set = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=download,
        transform=transform,
    )
    test_set = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=download,
        transform=transform,
    )

    generator = torch.Generator()
    if generator_state is None:
        generator.manual_seed(seed)
    else:
        generator.set_state(generator_state.cpu())
    common: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=generator,
        **common,
    )
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, test_loader, generator


def split_parameter_groups(
    model: nn.Module,
) -> Tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
    """Separate DEND+SOMA dynamics from the surrounding backbone."""

    dynamics = []
    backbone = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".dend." in name or ".soma." in name:
            dynamics.append(parameter)
        else:
            backbone.append(parameter)
    if not dynamics or not backbone:
        raise RuntimeError("Could not construct both optimizer parameter groups")
    return backbone, dynamics


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    neuron_learning_rate: float,
    weight_decay: float,
    neuron_weight_decay: float,
) -> torch.optim.Optimizer:
    backbone, dynamics = split_parameter_groups(model)
    return torch.optim.AdamW(
        [
            {
                "params": backbone,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "group_name": "backbone",
            },
            {
                "params": dynamics,
                "lr": neuron_learning_rate,
                "weight_decay": neuron_weight_decay,
                "group_name": "dend_soma",
            },
        ]
    )


def _topk_correct(logits: Tensor, target: Tensor, topk=(1, 5)) -> Sequence[int]:
    max_k = min(max(topk), logits.shape[1])
    predictions = logits.topk(max_k, dim=1).indices.transpose(0, 1)
    correct = predictions.eq(target.reshape(1, -1))
    return [
        int(correct[: min(k, max_k)].reshape(-1).sum().item()) for k in topk
    ]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    log_interval: int,
    max_batches: int,
    logger: logging.Logger,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    start = time.perf_counter()
    total_loss = 0.0
    total_samples = 0
    correct1 = 0
    correct5 = 0

    grad_context = torch.enable_grad() if training else torch.inference_mode()
    with grad_context:
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with amp_context:
                logits = model(inputs)
                loss = criterion(logits, targets)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = targets.shape[0]
            batch_correct1, batch_correct5 = _topk_correct(logits, targets)
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
            correct1 += batch_correct1
            correct5 += batch_correct5

            if log_interval > 0 and (batch_index + 1) % log_interval == 0:
                phase = "train" if training else "eval"
                logger.info(
                    "%s batch %d/%d loss=%.4f top1=%.2f%%",
                    phase,
                    batch_index + 1,
                    len(loader),
                    total_loss / total_samples,
                    100.0 * correct1 / total_samples,
                )

    if total_samples == 0:
        raise RuntimeError("The dataloader produced no samples")
    return EpochMetrics(
        loss=total_loss / total_samples,
        top1=100.0 * correct1 / total_samples,
        top5=100.0 * correct5 / total_samples,
        samples=total_samples,
        seconds=time.perf_counter() - start,
    )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "train.log", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def validate_device_compatibility(
    device: torch.device,
    dend_soma_config: DendSomaConfig,
) -> None:
    if device.type == "mps" and (
        dend_soma_config.dend_backend == "fft"
        or dend_soma_config.psn_backend == "fft"
    ):
        raise ValueError(
            "PyTorch MPS does not support the FFT kernels required by the "
            "selected DEND/SOMA backends. Use --device cpu/cuda. For a new "
            "MPS run, select --dend-backend gemm and --psn-backend gemm/conv; "
            "an FFT checkpoint keeps its stored backends when resumed."
        )

#  python train_scifar10_pixel.py --psn-exp-init --device cuda:3 --lr 0.01 --weight-decay 0.01  --soma-type psn_integer_ssf  --resume  logs/pixel_cifar10_dend_soma/masked_sliding_psn_seed1111_20260730-100804/checkpoint_best.pt --eval-only            --dropout 0.0      --neuron-lr 1e-3 --neuron-weight-decay 0
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train length-1024 pixel-by-pixel CIFAR-10 with the PMSN paper "
            "backbone and local DEND+SOMA neurons."
        )
    )
    parser.add_argument("--data-path", default="/data2/hyx/ViT-dend/data/cifar10", help="CIFAR-10 root")
    parser.add_argument("--output-dir", default="/data3/hyx/logs/pixel_cifar10_dend_soma")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="checkpoint path")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0", help="auto, cpu, mps, cuda, cuda:N")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="training epochs (default: 200; resume inherits checkpoint value)",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--neuron-lr",
        type=float,
        default=None,
        help="optional DEND+SOMA LR override; defaults to --lr",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--neuron-weight-decay",
        type=float,
        default=None,
        help="optional DEND+SOMA weight decay override; defaults to --weight-decay",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=300)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm", choices=["BN", "LN", "None"], default="BN")
    parser.add_argument(
        "--prenorm", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument(
        "--soma-type",
        choices=["masked_sliding_psn", "psn_integer_ssf"],
        default="masked_sliding_psn",
    )
    parser.add_argument("--psn-order", type=int, default=CIFAR10_SEQUENCE_LENGTH)
    parser.add_argument("--psn-backend", choices=["gemm", "conv", "fft"], default="fft")
    parser.add_argument("--psn-exp-init", action="store_true")
    parser.add_argument(
        "--integer-psn-bias",
        type=float,
        default=0.0,
        help="raw additive membrane bias for PSNIntergerSoma_ssf",
    )
    parser.add_argument("--ssf-thre", type=int, default=4)

    parser.add_argument("--num-branches", type=int, default=8)
    parser.add_argument("--compartments-per-branch", type=int, default=4)
    parser.add_argument("--branch-degree", type=int, default=1)
    parser.add_argument(
        "--branch-assignment", choices=["cyclic", "window"], default="cyclic"
    )
    parser.add_argument("--dend-backend", choices=["gemm", "fft"], default="fft")
    parser.add_argument(
        "--dend-merge-norm", choices=["sqrt", "mean", "sum"], default="mean"
    )
    parser.add_argument(
        "--learn-edge-gain", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--learn-comp-gain", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--activation-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "batch_size",
        "d_model",
        "n_layers",
        "psn_order",
        "num_branches",
        "compartments_per_branch",
        "branch_degree",
        "ssf_thre",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.lr <= 0.0 or (args.neuron_lr is not None and args.neuron_lr <= 0.0):
        raise ValueError("learning rates must be positive")
    if args.weight_decay < 0.0 or (
        args.neuron_weight_decay is not None and args.neuron_weight_decay < 0.0
    ):
        raise ValueError("weight decay cannot be negative")


def _make_configs(
    args: argparse.Namespace,
) -> Tuple[ModelConfig, DendSomaConfig]:
    model_config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prenorm=args.prenorm,
        norm=args.norm,
    )
    dend_soma_config = DendSomaConfig(
        soma_type=args.soma_type,
        psn_order=args.psn_order,
        psn_backend=args.psn_backend,
        psn_exp_init=args.psn_exp_init,
        integer_psn_bias=args.integer_psn_bias,
        ssf_thre=args.ssf_thre,
        num_branches=args.num_branches,
        compartments_per_branch=args.compartments_per_branch,
        branch_degree=args.branch_degree,
        branch_assignment=args.branch_assignment,
        dend_backend=args.dend_backend,
        dend_merge_norm=args.dend_merge_norm,
        learn_edge_gain=args.learn_edge_gain,
        learn_comp_gain=args.learn_comp_gain,
        activation_checkpoint=args.activation_checkpoint,
    )
    return model_config, dend_soma_config


def _run_directory(args: argparse.Namespace) -> Path:
    if args.resume:
        return Path(args.resume).expanduser().resolve().parent
    run_name = args.run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{args.soma_type}_seed{args.seed}_{timestamp}"
    return Path(args.output_dir).expanduser().resolve() / run_name


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    if device.type == "mps" and args.amp:
        raise ValueError("--amp is currently supported only on CUDA")
    amp_enabled = bool(args.amp and device.type == "cuda")

    run_dir = _run_directory(args)
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Use a new --run-name "
            "or resume an existing checkpoint."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir)
    checkpoint_data: Optional[Dict[str, Any]] = None
    if args.resume:
        checkpoint_data = torch.load(args.resume, map_location="cpu",weights_only=False)

    if checkpoint_data is not None and {
        "model_config",
        "dend_soma_config",
    }.issubset(checkpoint_data):
        model_config = ModelConfig(**checkpoint_data["model_config"])
        dend_soma_config = DendSomaConfig(**checkpoint_data["dend_soma_config"])
        logger.info("using architecture stored in %s", args.resume)
    else:
        model_config, dend_soma_config = _make_configs(args)
    stored_total_epochs = None
    if checkpoint_data is not None:
        stored_total_epochs = checkpoint_data.get("total_epochs")
        if stored_total_epochs is None:
            stored_total_epochs = checkpoint_data.get("scheduler", {}).get("T_max")
    if (
        stored_total_epochs is not None
        and args.epochs is not None
        and args.epochs != int(stored_total_epochs)
    ):
        raise ValueError(
            "--epochs cannot change when resuming a cosine-scheduled run: "
            f"checkpoint={stored_total_epochs}, requested={args.epochs}"
        )
    total_epochs = (
        args.epochs
        if args.epochs is not None
        else int(stored_total_epochs) if stored_total_epochs is not None else 200
    )
    validate_device_compatibility(device, dend_soma_config)
    model = PixelCIFAR10DendSomaModel(model_config, dend_soma_config).to(device)

    optimizer = build_optimizer(
        model,
        learning_rate=args.lr,
        neuron_learning_rate=(args.lr if args.neuron_lr is None else args.neuron_lr),
        weight_decay=args.weight_decay,
        neuron_weight_decay=(
            args.weight_decay
            if args.neuron_weight_decay is None
            else args.neuron_weight_decay
        ),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_top1 = float("-inf")
    if checkpoint_data is not None:
        model.load_state_dict(checkpoint_data["model"])
        if not args.eval_only:
            optimizer.load_state_dict(checkpoint_data["optimizer"])
            scheduler.load_state_dict(checkpoint_data["scheduler"])
            if amp_enabled and checkpoint_data.get("scaler"):
                scaler.load_state_dict(checkpoint_data["scaler"])
        start_epoch = int(checkpoint_data.get("epoch", 0))
        best_top1 = float(checkpoint_data.get("best_top1", float("-inf")))
        logger.info("resumed %s at epoch %d", args.resume, start_epoch)

    train_loader, test_loader, train_generator = build_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        generator_state=(
            checkpoint_data.get("data_generator_state")
            if checkpoint_data is not None
            else None
        ),
    )
    if checkpoint_data is not None and "rng_state" in checkpoint_data:
        restore_rng_state(checkpoint_data["rng_state"])

    total_parameters, trainable_parameters = count_parameters(model)
    if args.amp and not amp_enabled:
        logger.warning("--amp requested on %s; AMP is enabled only for CUDA", device)
    logger.info("device=%s amp=%s", device, amp_enabled)
    logger.info(
        "parameters total=%d trainable=%d",
        total_parameters,
        trainable_parameters,
    )
    logger.info("model_config=%s", model_config)
    logger.info("dend_soma_config=%s", dend_soma_config)

    config_path = run_dir / "config.json"
    if not config_path.exists():
        with config_path.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "arguments": vars(args),
                    "total_epochs": total_epochs,
                    "model": asdict(model_config),
                    "dend_soma": asdict(dend_soma_config),
                },
                stream,
                indent=2,
                sort_keys=True,
            )

    if args.eval_only:
        evaluation = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            scaler=scaler,
            amp_enabled=amp_enabled,
            log_interval=args.log_interval,
            max_batches=args.max_eval_batches,
            logger=logger,
        )
        logger.info("eval %s", evaluation)
        return 0

    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(start_epoch, total_epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            log_interval=args.log_interval,
            max_batches=args.max_train_batches,
            logger=logger,
        )
        eval_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            scaler=scaler,
            amp_enabled=amp_enabled,
            log_interval=args.log_interval,
            max_batches=args.max_eval_batches,
            logger=logger,
        )
        epoch_learning_rates = [group["lr"] for group in optimizer.param_groups]
        scheduler.step()

        is_best = eval_metrics.top1 > best_top1
        best_top1 = max(best_top1, eval_metrics.top1)
        checkpoint_state = {
            "epoch": epoch + 1,
            "total_epochs": total_epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_top1": best_top1,
            "model_config": asdict(model_config),
            "dend_soma_config": asdict(dend_soma_config),
            "data_generator_state": train_generator.get_state(),
            "rng_state": capture_rng_state(),
        }
        save_checkpoint(checkpoint_state, run_dir / "checkpoint_latest.pt")
        if is_best:
            save_checkpoint(checkpoint_state, run_dir / "checkpoint_best.pt")

        record = {
            "epoch": epoch + 1,
            "learning_rates": epoch_learning_rates,
            "train": asdict(train_metrics),
            "eval": asdict(eval_metrics),
            "best_top1": best_top1,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        logger.info(
            "epoch=%d/%d train_loss=%.4f train_top1=%.2f%% "
            "eval_loss=%.4f eval_top1=%.2f%% best_top1=%.2f%%",
            epoch + 1,
            total_epochs,
            train_metrics.loss,
            train_metrics.top1,
            eval_metrics.loss,
            eval_metrics.top1,
            best_top1,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())