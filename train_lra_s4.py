"""Train S4-family models on official-S4 Long Range Arena data.

Dataset preprocessing and undisclosed training details follow the local S4
repository. Base learning rate, base weight decay, batch size, and epoch count
follow MMDEND Appendix C, Table 7. DEND and SOMA use dedicated optimizer
groups. By default the activation immediately after FFTConv in each S4 block
uses the project's DEND+SOMA module. The final activation can instead be
replaced independently, while the residual path retains its S4 definition.

The optional ``spikingssm_pathx`` recipe is isolated from that existing route.
It follows the local SDN repository's PathX S4D model and training config, with
only the SDN activation replaced by this project's DEND+SOMA module.

The ``s4_v3`` recipe is a separate, config-faithful route for the six archived
``old/v3-s4-*.yaml`` experiments. It restores their model, decoder, data,
optimizer, scheduler, batch, epoch, and seed settings without changing the
existing MMDEND/V4 route.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from lra_dataset import (
    canonicalize_lra_task,
    get_s4_lra_data,
    get_s4_v3_lra_data,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# MMDEND Appendix C, Table 7. These four values are intentionally not taken
# from another benchmark implementation.
MMDEND_TRAINING_PRESETS: Dict[str, Dict[str, float | int]] = {
    "aan": {"lr": 0.01, "weight_decay": 0.05, "batch_size": 64, "epochs": 20},
    "cifar": {"lr": 0.01, "weight_decay": 0.05, "batch_size": 50, "epochs": 200},
    "imdb": {"lr": 0.01, "weight_decay": 0.05, "batch_size": 16, "epochs": 32},
    "pathfinder": {
        "lr": 0.004,
        "weight_decay": 0.05,
        "batch_size": 64,
        "epochs": 200,
    },
    "listops": {"lr": 0.01, "weight_decay": 0.05, "batch_size": 32, "epochs": 40},
    "pathx": {"lr": 0.001, "weight_decay": 0.05, "batch_size": 16, "epochs": 50},
}

# SDN/configs/experiment/spikingssm/pathx.yaml. These values deliberately
# remain separate from the MMDEND presets above.
SPIKINGSSM_PATHX_TRAINING_PRESET: Dict[str, float | int] = {
    "lr": 0.001,
    "weight_decay": 0.01,
    "batch_size": 16,
    "epochs": 50,
}

SPIKINGSSM_PATHX_SCHEDULER_PRESET: Dict[str, int] = {
    "num_training_steps": 500000,
    "num_warmup_steps": 50000,
}

# state-spaces/s4 configs/experiment/lra/s4-*.yaml at the local S4 revision.
S4_SCHEDULER_PRESETS: Dict[str, Dict[str, int]] = {
    "aan": {"num_training_steps": 50000, "num_warmup_steps": 5000},
    "cifar": {"num_training_steps": 180000, "num_warmup_steps": 18000},
    "imdb": {"num_training_steps": 50000, "num_warmup_steps": 5000},
    "pathfinder": {"num_training_steps": 500000, "num_warmup_steps": 50000},
    "listops": {"num_training_steps": 120000, "num_warmup_steps": 12000},
    "pathx": {"num_training_steps": 500000, "num_warmup_steps": 50000},
}

S4_TRAINING_SEEDS = {
    "aan": 3333,
    "cifar": 2222,
    "imdb": 3333,
    "pathfinder": 3333,
    "listops": 3333,
    "pathx": 3333,
}


# state-spaces/s4 configs/experiment/lra/old/v3-s4-*.yaml. These values are
# deliberately duplicated instead of inheriting MMDEND_TRAINING_PRESETS: the
# V3 ListOps batch size and Pathfinder weight decay are different.
S4_V3_TRAINING_PRESETS: Dict[str, Dict[str, float | int]] = {
    "aan": {"lr": 0.01, "weight_decay": 0.05, "batch_size": 64, "epochs": 20},
    "cifar": {"lr": 0.01, "weight_decay": 0.05, "batch_size": 50, "epochs": 200},
    "imdb": {"lr": 0.01, "weight_decay": 0.05, "batch_size": 16, "epochs": 32},
    "pathfinder": {
        "lr": 0.004,
        "weight_decay": 0.03,
        "batch_size": 64,
        "epochs": 200,
    },
    "listops": {"lr": 0.01, "weight_decay": 0.05, "batch_size": 50, "epochs": 40},
    "pathx": {"lr": 0.001, "weight_decay": 0.05, "batch_size": 16, "epochs": 50},
}

S4_V3_SCHEDULER_PRESETS: Dict[str, Dict[str, int]] = {
    # V3 inherits 1000 warmup steps from scheduler/cosine_warmup.yaml when the
    # experiment file does not override that field.
    "aan": {"num_training_steps": 50000, "num_warmup_steps": 2500},
    "cifar": {"num_training_steps": 180000, "num_warmup_steps": 900},
    "imdb": {"num_training_steps": 50000, "num_warmup_steps": 1000},
    "pathfinder": {"num_training_steps": 500000, "num_warmup_steps": 2500},
    "listops": {"num_training_steps": 80000, "num_warmup_steps": 1000},
    "pathx": {"num_training_steps": 500000, "num_warmup_steps": 50000},
}

S4_V3_TRAINING_SEEDS = {task: 2222 for task in S4_V3_TRAINING_PRESETS}


def load_s4_lra_module():
    module_name = "_vit_dend_s4_lra_shared"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(__file__).resolve().parent / "model" / "s4_lra.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load S4 LRA module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_spikingssm_pathx_module():
    path = Path(__file__).resolve().parent / "model" / "spikingssm_pathx.py"
    spec = importlib.util.spec_from_file_location("spikingssm_pathx_file", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SpikingSSM PathX module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not distributed_is_initialized() or dist.get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def setup_distributed(args) -> tuple[bool, int, int, int, torch.device]:
    """Initialize one-process-per-GPU DDP when launched with torchrun."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank or 0))
    distributed = world_size > 1

    if distributed:
        requested_device = torch.device(args.device)
        if requested_device.type != "cuda":
            raise ValueError("Multi-GPU LRA training requires --device cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun requested CUDA DDP, but CUDA is unavailable")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} "
                "visible CUDA devices were found"
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        requested_device = torch.device(args.device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            print(f"CUDA is unavailable; falling back from {requested_device} to cpu")
            device = torch.device("cpu")
        else:
            device = requested_device

    return distributed, rank, local_rank, world_size, device


def _backbone_no_weight_decay_parameter_ids(model: nn.Module) -> set[int]:
    """Find the backbone parameters excluded from decay by official S4 hooks."""

    normalization_modules = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
        nn.LocalResponseNorm,
    )
    blacklist_modules = (nn.Embedding, *normalization_modules)

    roots = []
    if hasattr(model, "blocks"):
        roots.append(model.blocks)
    if hasattr(model, "final_norm"):
        roots.append(model.final_norm)
    if hasattr(model, "backbone"):
        roots.append(model.backbone)

    no_weight_decay_ids = set()
    for root in roots:
        for module in root.modules():
            for local_name, parameter in module.named_parameters(recurse=False):
                if (
                    local_name.endswith("bias")
                    or getattr(parameter, "_no_weight_decay", False)
                    or isinstance(module, blacklist_modules)
                ):
                    no_weight_decay_ids.add(id(parameter))
    return no_weight_decay_ids


def setup_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    *,
    dend_lr: float = 1e-3,
    dend_weight_decay: float = 0.0,
    soma_lr: float = 5e-4,
    soma_weight_decay: float = 0.0,
    dedicated_dend_soma_groups: bool = True,
    exclude_bias_norm_from_weight_decay: bool = True,
) -> optim.Optimizer:
    """Create AdamW groups while honoring official S4 ``_optim`` hooks."""

    hyperparameters = {
        "lr": lr,
        "weight_decay": weight_decay,
        "dend_lr": dend_lr,
        "dend_weight_decay": dend_weight_decay,
        "soma_lr": soma_lr,
        "soma_weight_decay": soma_weight_decay,
    }
    for name, value in hyperparameters.items():
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    def belongs_to(name: str, component: str) -> bool:
        return component in name.split(".")

    dend_parameters = (
        [
            parameter
            for name, parameter in named_parameters
            if belongs_to(name, "dend")
        ]
        if dedicated_dend_soma_groups
        else []
    )
    soma_parameters = (
        [
            parameter
            for name, parameter in named_parameters
            if belongs_to(name, "soma")
        ]
        if dedicated_dend_soma_groups
        else []
    )
    component_ids = {
        id(parameter) for parameter in dend_parameters + soma_parameters
    }

    s4_parameters = [
        parameter
        for _, parameter in named_parameters
        if id(parameter) not in component_ids and hasattr(parameter, "_optim")
    ]
    backbone_no_weight_decay_ids = (
        _backbone_no_weight_decay_parameter_ids(model)
        if exclude_bias_norm_from_weight_decay
        else set()
    )
    base_parameters = [
        parameter
        for _, parameter in named_parameters
        if id(parameter) not in component_ids
        and id(parameter) not in backbone_no_weight_decay_ids
        and not hasattr(parameter, "_optim")
    ]
    base_no_weight_decay_parameters = [
        parameter
        for _, parameter in named_parameters
        if id(parameter) not in component_ids
        and id(parameter) in backbone_no_weight_decay_ids
        and not hasattr(parameter, "_optim")
    ]

    parameter_groups = []
    if base_parameters:
        parameter_groups.append(
            {"params": base_parameters, "group_name": "base"}
        )
    if base_no_weight_decay_parameters:
        parameter_groups.append(
            {
                "params": base_no_weight_decay_parameters,
                "group_name": "base_no_weight_decay",
                "weight_decay": 0.0,
            }
        )

    s4_hyperparameters = []
    seen_s4_hyperparameters = set()
    for parameter in s4_parameters:
        custom = getattr(parameter, "_optim")
        key = frozenset(custom.items())
        if key not in seen_s4_hyperparameters:
            seen_s4_hyperparameters.add(key)
            s4_hyperparameters.append(custom)
    for index, custom in enumerate(s4_hyperparameters):
        parameters = [
            parameter
            for parameter in s4_parameters
            if getattr(parameter, "_optim") == custom
        ]
        parameter_groups.append(
            {
                "params": parameters,
                "group_name": f"s4_custom_{index}",
                **custom,
            }
        )

    if dend_parameters:
        parameter_groups.append(
            {
                "params": dend_parameters,
                "group_name": "dend",
                "lr": dend_lr,
                "weight_decay": dend_weight_decay,
            }
        )
    if soma_parameters:
        parameter_groups.append(
            {
                "params": soma_parameters,
                "group_name": "soma",
                "lr": soma_lr,
                "weight_decay": soma_weight_decay,
            }
        )

    assigned_ids = [
        id(parameter)
        for group in parameter_groups
        for parameter in group["params"]
    ]
    expected_ids = {id(parameter) for _, parameter in named_parameters}
    if len(assigned_ids) != len(set(assigned_ids)):
        raise RuntimeError("A parameter was assigned to more than one optimizer group")
    if set(assigned_ids) != expected_ids:
        raise RuntimeError("Some trainable parameters were not assigned to an optimizer group")

    defaults = {"lr": lr, "weight_decay": weight_decay, "betas": (0.9, 0.999)}
    optimizer = optim.AdamW(parameter_groups, **defaults)

    if is_main_process():
        for index, group in enumerate(optimizer.param_groups):
            parameter_count = sum(parameter.numel() for parameter in group["params"])
            print(
                f"Optimizer group {index} ({group['group_name']}): "
                f"{len(group['params'])} tensors {parameter_count} parameters "
                f"lr={group['lr']} weight_decay={group['weight_decay']}"
            )

    return optimizer


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    steps_per_epoch: int,
    num_training_steps: Optional[int],
    num_warmup_steps: Optional[int],
):
    if scheduler_name == "none":
        return None, "epoch"
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs), "epoch"

    total_steps = num_training_steps or epochs * steps_per_epoch
    warmup_steps = num_warmup_steps
    if warmup_steps is None:
        warmup_steps = int(0.1 * total_steps)

    # Exact lambda used by transformers.get_cosine_schedule_with_warmup,
    # which is the scheduler registered by the official S4 pipeline.
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"


def unpack_batch(batch, device: torch.device):
    if len(batch) == 3:
        inputs, targets, extra = batch
        lengths = extra.get("lengths")
    else:
        inputs, targets = batch
        lengths = None

    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if lengths is not None:
        lengths = lengths.to(device, non_blocking=True)
    return inputs, targets, lengths


def synchronize_sequence_length(inputs: torch.Tensor) -> torch.Tensor:
    """Pad variable-length token batches to one shared DDP sequence length."""

    if not distributed_is_initialized() or inputs.dim() < 2:
        return inputs
    local_length = torch.tensor(inputs.size(1), device=inputs.device, dtype=torch.long)
    dist.all_reduce(local_length, op=dist.ReduceOp.MAX)
    global_length = int(local_length.item())
    if inputs.size(1) < global_length:
        padding_shape = (
            inputs.size(0),
            global_length - inputs.size(1),
            *inputs.shape[2:],
        )
        inputs = torch.cat([inputs, inputs.new_zeros(padding_shape)], dim=1)
    return inputs


def run_epoch(
    loader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler,
    scheduler_interval: str,
    scaler: Optional[amp.GradScaler],
    device: torch.device,
    train: bool,
    print_freq: int,
    epoch: int,
) -> Tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    phase = "train" if train else "eval"

    iterator = tqdm(
        enumerate(loader),
        total=len(loader),
        leave=False,
        disable=not is_main_process(),
    )
    for batch_idx, batch in iterator:
        inputs, targets, lengths = unpack_batch(batch, device)
        inputs = synchronize_sequence_length(inputs)

        with torch.set_grad_enabled(train):
            with amp.autocast(enabled=scaler is not None):
                logits = model(inputs, lengths=lengths)
                loss = criterion(logits, targets)

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                if scheduler is not None and scheduler_interval == "step":
                    scheduler.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += batch_size

        if print_freq > 0 and (
            batch_idx % print_freq == 0 or batch_idx + 1 == len(loader)
        ):
            avg_loss = total_loss / max(total_seen, 1)
            avg_acc = 100.0 * total_correct / max(total_seen, 1)
            iterator.set_description(
                f"{phase} epoch={epoch} batch={batch_idx + 1}/{len(loader)} "
                f"loss={avg_loss:.4f} acc={avg_acc:.2f}"
            )

    totals = torch.tensor(
        [total_loss, float(total_correct), float(total_seen)],
        device=device,
        dtype=torch.float64,
    )
    if distributed_is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    total_loss, total_correct, total_seen = totals.tolist()

    if total_seen == 0:
        raise RuntimeError(
            "The dataloader yielded no samples. With official S4 drop_last=True, "
            "the selected subset must contain at least one full batch."
        )
    return total_loss / total_seen, 100.0 * total_correct / total_seen


def save_checkpoint(state: dict, output_dir: Path, is_best: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "checkpoint.pth.tar"
    torch.save(state, latest)
    if is_best:
        torch.save(state, output_dir / "model_best.pth.tar")

# python train_lra_s4.py --task imdb --device cuda:0 --soma-lr 0.001 --dend-lr 0.001 --soma-type psn_integer_ssf --resume /data2/hyx/ViT-dend2/exp/lra-new-cifar-dend_soma-gelu-2026-09-23-04-02-51/checkpoint.pth.tar --output-dir /data2/hyx/ViT-dend2/exp/lra-new-cifar-dend_soma-gelu-2026-09-23-04-02-51
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8  train_lra_s4.py --task pathx  --soma-lr 0.001 --dend-lr 0.001 --soma-type psn_integer_ssf --resume /data2/hyx/ViT-dend2/exp/lra-new-pathx-dend_soma-gelu-2026-09-20-03-52-24/checkpoint.pth.tar --output-dir /data2/hyx/ViT-dend2/exp/lra-new-pathx-dend_soma-gelu-2026-09-20-03-52-24
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc-per-node=4 train_lra_s4.py --task aan --device cuda --soma-lr 0.001 --dend-lr 0.001 --soma-type psn_integer_ssf --recipe s4_v3 --batch-size 16      --lr 0.01   --dend-compartments 2
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train S4-LRA with optional DEND+SOMA activations."
    )
    parser.add_argument(
        "--task",
        default="listops",
        choices=[
            "aan",
            "retrieval",
            "cifar",
            "image",
            "imdb",
            "text",
            "pathfinder",
            "listops",
            "pathx",
        ],
    )
    parser.add_argument(
        "--recipe",
        default="mmdend_s4",
        choices=["mmdend_s4", "s4_v3", "spikingssm_pathx"],
        help=(
            "mmdend_s4 preserves the existing model/training path; "
            "s4_v3 follows state-spaces/s4 old/v3-s4-*.yaml; "
            "spikingssm_pathx uses the local SDN repository's exact PathX "
            "S4D macro-model and training hyperparameters, replacing only its "
            "neuron activation with DEND+SOMA."
        ),
    )
    parser.add_argument(
        "--root",
        default="/data2/hyx/ViT-dend/data/lra_release",
        help="Root containing raw IMDB, CIFAR-10, ListOps, AAN, and Pathfinder data.",
    )
    parser.add_argument(
        "--s4-root",
        default="/data2/hyx/s4",
        help="Official S4 repo root to add to PYTHONPATH.", ##
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help=(
            "Torch device, for example cuda, cuda:4, or cpu. Under torchrun, "
            "use cuda and LOCAL_RANK selects each process GPU."
        ),
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dist-backend",
        default="nccl",
        choices=["nccl", "gloo"],
        help="torch.distributed backend used when launched with torchrun.",
    )
    parser.add_argument("--backend", default="official", choices=["official", "fallback", "auto"])
    parser.add_argument("--activation", default="dend_soma", choices=["dend_soma", "standard"])
    parser.add_argument(
        "--dend-soma-target",
        default="gelu",
        choices=["gelu", "final_act", "both"],
        help=(
            "S4Block activation replaced by DEND+SOMA: the post-FFTConv GELU, "
            "the final_act after output projection, or both. Replacing final_act "
            "uses an H-to-H projection instead of GLU's H-to-2H projection. "
            "This option applies to the mmdend_s4 and s4_v3 recipes; "
            "spikingssm_pathx always uses the SDN neuron position."
        ),
    )
    parser.add_argument("--output-dir", default="", help="Directory for args/checkpoints. Default creates exp/lra-*.")
    parser.add_argument("--resume", default="", help="Resume from checkpoint.pth.tar/model_best.pth.tar.")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "For mmdend_s4 this is the global batch size and is divided across "
            "torchrun ranks. For s4_v3 and spikingssm_pathx it is the official "
            "S4/Lightning DataLoader's per-rank batch size. For AAN it counts "
            "document pairs, not individual documents."
        ),
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", "--wd", dest="weight_decay", type=float, default=None)
    parser.add_argument("--scheduler", default="cosine-warmup", choices=["cosine-warmup", "cosine", "none"])
    parser.add_argument("--num-training-steps", type=int, default=None)
    parser.add_argument("--num-warmup-steps", type=int, default=None)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-state", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument(
        "--eval-test-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Evaluate the test loader after every validation epoch. Defaults to "
            "disabled for strict s4_v3 and preserves the existing enabled "
            "behavior for other recipes."
        ),
    )
    parser.add_argument("--zero-pad-embedding", action="store_true", help="Use padding_idx=0 in token embedding.")

    parser.add_argument("--dend-branches", type=int, default=8)
    parser.add_argument("--dend-compartments", type=int, default=4)
    parser.add_argument("--dend-branch-degree", type=int, default=1)
    parser.add_argument(
        "--dend-lr",
        type=float,
        default=1e-3,
        help=(
            "Peak DEND learning rate when dedicated groups are enabled. "
            "spikingssm_pathx ignores this value."
        ),
    )
    parser.add_argument(
        "--dend-weight-decay",
        "--dend-wd",
        dest="dend_weight_decay",
        type=float,
        default=0.0,
        help=(
            "DEND weight decay when dedicated groups are enabled. "
            "spikingssm_pathx ignores this value."
        ),
    )
    parser.add_argument(
        "--dend-integration-backend",
        default="fft",
        choices=["gemm", "fft"],
    )
    parser.add_argument(
        "--soma-type",
        default="masked_sliding_psn",
        choices=["masked_sliding_psn", "psn_integer_ssf"],
        help=(
            "Soma used after the channel-preserving dendrite at each selected "
            "S4Block activation target."
        ),
    )
    parser.add_argument(
        "--soma-lr",
        type=float,
        default=5e-4,
        help=(
            "Peak SOMA learning rate when dedicated groups are enabled. "
            "spikingssm_pathx ignores this value."
        ),
    )
    parser.add_argument(
        "--soma-weight-decay",
        "--soma-wd",
        dest="soma_weight_decay",
        type=float,
        default=0.0,
        help=(
            "SOMA weight decay when dedicated groups are enabled. "
            "spikingssm_pathx ignores this value."
        ),
    )
    parser.add_argument(
        "--soma-psn-order",
        type=int,
        default=None,
        help="Temporal window order; defaults to the loaded dataset sequence length.",
    )
    parser.add_argument(
        "--soma-psn-backend",
        default="fft",
        choices=["gemm", "conv", "fft"],
    )
    parser.add_argument(
        "--soma-psn-exp-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use exponential initialization for the selected PSN when supported.",
    )
    parser.add_argument(
        "--soma-psn-threshold-init",
        type=float,
        default=0.0,
        help="Initial temporal bias for psn_integer_ssf; ignored by masked_sliding_psn.",
    )
    parser.add_argument(
        "--soma-ssf-thre",
        type=int,
        default=4,
        help="Signed SSF clipping level for psn_integer_ssf.",
    )
    parser.add_argument(
        "--dend-soma-activation-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recompute each DEND+SOMA activation during backward to reduce "
            "long-sequence activation memory."
        ),
    )
    parser.add_argument(
        "--dedicated-dend-soma-optimizer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Put DEND and SOMA in their dedicated LR/WD groups. Defaults to "
            "enabled for DEND+SOMA under mmdend_s4 and s4_v3."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    s4_root = Path(args.s4_root).expanduser()
    if s4_root.is_dir() and str(s4_root) not in sys.path:
        sys.path.insert(0, str(s4_root))

    s4_lra = load_s4_lra_module()
    task_key = canonicalize_lra_task(args.task)
    args.task = task_key
    if args.recipe == "spikingssm_pathx":
        if task_key != "pathx":
            raise ValueError(
                "The spikingssm_pathx recipe is defined only for --task pathx"
            )
        if args.activation != "dend_soma":
            raise ValueError(
                "The spikingssm_pathx recipe replaces the SDN neuron with "
                "DEND+SOMA and therefore requires --activation dend_soma"
            )
        if args.backend != "official":
            raise ValueError(
                "The spikingssm_pathx recipe requires --backend official so it "
                "can use the same SSMKernelDiag implementation as SDN"
            )
        training_preset = SPIKINGSSM_PATHX_TRAINING_PRESET
        scheduler_preset = SPIKINGSSM_PATHX_SCHEDULER_PRESET
        seed_preset = S4_TRAINING_SEEDS
    elif args.recipe == "s4_v3":
        training_preset = S4_V3_TRAINING_PRESETS[task_key]
        scheduler_preset = S4_V3_SCHEDULER_PRESETS[task_key]
        seed_preset = S4_V3_TRAINING_SEEDS
    else:
        training_preset = MMDEND_TRAINING_PRESETS[task_key]
        scheduler_preset = S4_SCHEDULER_PRESETS[task_key]
        seed_preset = S4_TRAINING_SEEDS

    if args.eval_test_every_epoch is None:
        args.eval_test_every_epoch = True

    args.epochs = (
        int(training_preset["epochs"]) if args.epochs is None else args.epochs
    )
    args.batch_size = (
        int(training_preset["batch_size"])
        if args.batch_size is None
        else args.batch_size
    )
    args.lr = float(training_preset["lr"]) if args.lr is None else args.lr
    args.weight_decay = (
        float(training_preset["weight_decay"])
        if args.weight_decay is None
        else args.weight_decay
    )
    if args.num_training_steps is None:
        args.num_training_steps = scheduler_preset["num_training_steps"]
    if args.num_warmup_steps is None:
        args.num_warmup_steps = scheduler_preset["num_warmup_steps"]
    if args.seed is None:
        args.seed = seed_preset[task_key]

    distributed, rank, local_rank, world_size, device = setup_distributed(args)
    set_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    if distributed and args.recipe in {"s4_v3", "spikingssm_pathx"}:
        # Lightning leaves loader.batch_size unchanged on every DDP process.
        local_batch_size = args.batch_size
        effective_global_batch_size = args.batch_size * world_size
        args.batch_size_scope = "per_rank"
    elif distributed:
        if args.batch_size % world_size != 0:
            raise ValueError(
                f"Global batch size {args.batch_size} must be divisible by "
                f"world size {world_size}"
            )
        local_batch_size = args.batch_size // world_size
        effective_global_batch_size = args.batch_size
        args.batch_size_scope = "global"
    else:
        local_batch_size = args.batch_size
        effective_global_batch_size = args.batch_size
        args.batch_size_scope = "single_process"
    args.distributed = distributed
    args.world_size = world_size
    args.local_batch_size = local_batch_size
    args.effective_global_batch_size = effective_global_batch_size

    max_samples = {
        "train": args.max_train_samples,
        "val": args.max_val_samples,
        "test": args.max_test_samples,
    }
    max_samples = {k: v for k, v in max_samples.items() if v is not None}
    if distributed and not is_main_process():
        dist.barrier()
    data_builder = get_s4_v3_lra_data if args.recipe == "s4_v3" else get_s4_lra_data
    data = data_builder(
        task=task_key,
        root=args.root,
        s4_root=args.s4_root,
        batch_size=local_batch_size,
        num_workers=args.workers,
        max_samples=max_samples,
        max_len=args.max_len,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        distributed_seed=args.seed,
    )
    if distributed and is_main_process():
        dist.barrier()
    spec = data.spec
    loaders = data.loaders
    if args.activation == "dend_soma" and args.soma_psn_order is None:
        args.soma_psn_order = 4 #spec.sequence_length // 100 ##
        args.soma_psn_exp_init = True ##
    args.data_pipeline = (
        "official_s4_v3" if args.recipe == "s4_v3" else "official_s4_v4"
    )
    args.data_config_profile = data.config_profile
    if args.recipe == "spikingssm_pathx":
        args.model_source = "local SDN SpikingSSM PathX S4D-Lin implementation"
        args.training_hparams_source = (
            "SDN/configs/experiment/spikingssm/pathx.yaml"
        )
        args.activation_location = "SDN neuron position after S4D convolution"
        args.dedicated_dend_soma_optimizer = True
        args.exclude_bias_norm_from_weight_decay = True
    elif args.recipe == "s4_v3":
        args.model_source = (
            "official S4Block resolved from "
            "state-spaces/s4 old/v3-s4-*.yaml"
        )
        args.training_hparams_source = (
            "state-spaces/s4 configs/experiment/lra/old/v3-s4-*.yaml"
        )
        args.activation_location = args.dend_soma_target
        if args.dedicated_dend_soma_optimizer is None:
            args.dedicated_dend_soma_optimizer = args.activation == "dend_soma"
        if args.activation == "standard":
            args.dedicated_dend_soma_optimizer = False
        elif args.dedicated_dend_soma_optimizer:
            args.training_hparams_source += (
                " plus dedicated DEND/SOMA optimizer groups"
            )
        # V3 predates train.optimizer_param_grouping in configs/config.yaml.
        args.exclude_bias_norm_from_weight_decay = False
    else:
        args.model_source = "official S4Block with MMDEND activation placement"
        args.training_hparams_source = "MMDEND Appendix C Table 7"
        args.activation_location = args.dend_soma_target
        if args.dedicated_dend_soma_optimizer is None:
            args.dedicated_dend_soma_optimizer = args.activation == "dend_soma"
        if args.activation == "standard":
            args.dedicated_dend_soma_optimizer = False
        elif args.dedicated_dend_soma_optimizer:
            args.training_hparams_source += (
                " with dedicated DEND/SOMA optimizer groups"
            )
        args.exclude_bias_norm_from_weight_decay = True
    args.drop_last = True
    args.pin_memory = True
    args.validation_uses_test = data.validation_uses_test

    model_overrides = {}
    if args.d_model is not None:
        model_overrides["d_model"] = args.d_model
    if args.d_state is not None:
        model_overrides["d_state"] = args.d_state
    if args.n_layers is not None:
        model_overrides["n_layers"] = args.n_layers
    if args.dropout is not None:
        model_overrides["dropout"] = args.dropout
    dend_soma_overrides = {
        "dend_soma_num_branches": args.dend_branches,
        "dend_soma_compartments_per_branch": args.dend_compartments,
        "dend_soma_branch_degree": args.dend_branch_degree,
        "dend_soma_dend_backend": args.dend_integration_backend,
        "dend_soma_soma_type": args.soma_type,
        "dend_soma_psn_order": args.soma_psn_order,
        "dend_soma_psn_backend": args.soma_psn_backend,
        "dend_soma_psn_exp_init": args.soma_psn_exp_init,
        "dend_soma_psn_threshold_init": args.soma_psn_threshold_init,
        "dend_soma_ssf_thre": args.soma_ssf_thre,
        "dend_soma_activation_checkpoint": args.dend_soma_activation_checkpoint,
    }

    if args.recipe == "spikingssm_pathx":
        spikingssm_pathx = load_spikingssm_pathx_module()
        model = spikingssm_pathx.build_spikingssm_pathx_dend_soma(
            d_input=spec.d_input,
            d_output=spec.d_output,
            vocab_size=spec.vocab_size,
            sequence_length=spec.sequence_length,
            **model_overrides,
            **dend_soma_overrides,
        ).to(device)
    else:
        if args.activation == "dend_soma":
            model_overrides.update(dend_soma_overrides)
            model_overrides["dend_soma_activation_target"] = (
                args.dend_soma_target
            )
        if args.recipe == "s4_v3":
            builder = (
                s4_lra.build_dend_soma_s4_lra_v3
                if args.activation == "dend_soma"
                else s4_lra.build_standard_s4_lra_v3
            )
        else:
            builder = (
                s4_lra.build_dend_soma_s4_lra
                if args.activation == "dend_soma"
                else s4_lra.build_standard_s4_lra
            )
        embedding_padding_idx = (
            spec.padding_idx if args.zero_pad_embedding else None
        )
        model = builder(
            task_key,
            d_input=spec.d_input,
            d_output=spec.d_output,
            vocab_size=spec.vocab_size,
            backend=args.backend,
            padding_idx=embedding_padding_idx,
            **model_overrides,
        ).to(device)

    if args.recipe in {"s4_v3", "spikingssm_pathx"}:
        args.resolved_model_config = dict(model.config.__dict__)
    else:
        args.resolved_model_config = None

    # SDN retains ordinary per-rank BatchNorm from its reference pipeline.
    args.sync_batchnorm = distributed #and args.recipe in {"mmdend_s4", "s4_v3"}  ##
    if args.sync_batchnorm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = setup_optimizer(
        model,
        args.lr,
        args.weight_decay,
        dend_lr=args.dend_lr,
        dend_weight_decay=args.dend_weight_decay,
        soma_lr=args.soma_lr,
        soma_weight_decay=args.soma_weight_decay,
        dedicated_dend_soma_groups=args.dedicated_dend_soma_optimizer,
        exclude_bias_norm_from_weight_decay=(
            args.exclude_bias_norm_from_weight_decay
        ),
    )
    scheduler, scheduler_interval = build_scheduler(
        optimizer,
        args.scheduler,
        args.epochs,
        len(loaders["train"]),
        args.num_training_steps,
        args.num_warmup_steps,
    )
    scaler = amp.GradScaler() if args.amp and device.type == "cuda" else None

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            # The channel-preserving dendrite keeps inactive readout parameters
            # for interface compatibility, so DDP must tolerate unused tensors.
            find_unused_parameters=args.activation == "dend_soma",
        )
        set_seed(args.seed + rank)

    output_dir_value = args.output_dir or None
    if output_dir_value is None and is_main_process():
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        activation_label = args.activation
        if args.activation == "dend_soma":
            activation_label += (
                "-sdn-neuron"
                if args.recipe == "spikingssm_pathx"
                else f"-{args.dend_soma_target}"
            )
        run_name = (
            f"lra-new-{task_key}-{activation_label}-{stamp}"
            if args.recipe == "mmdend_s4"
            else f"lra-{args.recipe}-{task_key}-{activation_label}-{stamp}"
        )
        output_dir_value = str(Path("exp") / run_name)
    if distributed:
        output_dir_values = [output_dir_value]
        dist.broadcast_object_list(output_dir_values, src=0)
        output_dir_value = output_dir_values[0]
    if output_dir_value is None:
        raise RuntimeError("Failed to resolve the training output directory")
    output_dir = Path(output_dir_value)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    start_epoch = 0
    best_val = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device,weights_only=False)
        checkpoint_args = checkpoint.get("args") or {}
        checkpoint_recipe = checkpoint_args.get("recipe", "mmdend_s4")
        if checkpoint_recipe != args.recipe:
            raise ValueError(
                "Checkpoint recipe does not match this run: "
                f"{checkpoint_recipe!r} != {args.recipe!r}"
            )
        checkpoint_activation = checkpoint_args.get("activation")
        if (
            checkpoint_activation is not None
            and checkpoint_activation != args.activation
        ):
            raise ValueError(
                "Checkpoint activation mode does not match this run: "
                f"{checkpoint_activation!r} != {args.activation!r}"
            )
        if args.activation == "dend_soma" and args.recipe in {
            "mmdend_s4",
            "s4_v3",
        }:
            checkpoint_target = checkpoint_args.get("dend_soma_target", "gelu")
            if checkpoint_target != args.dend_soma_target:
                raise ValueError(
                    "Checkpoint DEND+SOMA target does not match this run: "
                    f"{checkpoint_target!r} != {args.dend_soma_target!r}"
                )
        unwrap_model(model).load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val = float(checkpoint.get("best_val", -1.0))
        if is_main_process():
            print(f"Resumed {args.resume} at epoch {start_epoch} best_val={best_val:.2f}")

    if is_main_process():
        with (output_dir / "args.json").open("w") as f:
            json.dump(vars(args), f, indent=2)

    if is_main_process():
        print(f"Task={task_key} spec={spec}")
        print(
            f"Recipe={args.recipe} device={device} backend={args.backend} "
            f"activation={args.activation}"
        )
        print(f"Model source={args.model_source}")
        if args.resolved_model_config is not None:
            print(f"Resolved model config={args.resolved_model_config}")
        if distributed:
            print(
                f"DDP world_size={world_size} "
                f"effective_global_batch_size={args.effective_global_batch_size} "
                f"local_batch_size={local_batch_size} "
                f"configured_batch_scope={args.batch_size_scope} "
                f"sync_batchnorm={args.sync_batchnorm}"
            )
        if args.activation == "dend_soma":
            print(
                "DEND+SOMA "
                f"target={args.activation_location} "
                f"branches={args.dend_branches} compartments={args.dend_compartments} "
                f"branch_degree={args.dend_branch_degree} "
                f"dend_backend={args.dend_integration_backend} "
                f"soma={args.soma_type} "
                f"psn_order={args.soma_psn_order} psn_backend={args.soma_psn_backend} "
                f"activation_checkpoint={args.dend_soma_activation_checkpoint}"
            )
            if args.dedicated_dend_soma_optimizer:
                print(
                    "DEND/SOMA optimizer "
                    f"dend_lr={args.dend_lr} dend_wd={args.dend_weight_decay} "
                    f"soma_lr={args.soma_lr} soma_wd={args.soma_weight_decay}"
                )
            else:
                weight_decay_detail = (
                    "with zero decay for biases and normalization parameters"
                    if args.exclude_bias_norm_from_weight_decay
                    else "with V3 weight decay on all non-S4 parameters"
                )
                print(
                    "DEND/SOMA optimizer=official S4 base grouping "
                    f"lr={args.lr} weight_decay={args.weight_decay} "
                    f"{weight_decay_detail}"
                )
        print(f"Output dir={output_dir}")
        print(
            f"Data pipeline={args.data_pipeline} "
            f"profile={args.data_config_profile} "
            f"data_dir={spec.data_dir} drop_last=True pin_memory=True "
            f"workers={args.workers}"
        )
        if data.validation_uses_test:
            print(
                "Validation protocol=official LRA IMDB: the test split is also used "
                "for validation/checkpoint selection"
            )
        print(
            f"Epochs={args.epochs} "
            f"effective_global_batch_size={args.effective_global_batch_size} "
            f"lr={args.lr} weight_decay={args.weight_decay} "
            f"train_batches={len(loaders['train'])} scheduler={args.scheduler}/{scheduler_interval} "
            f"warmup_steps={args.num_warmup_steps} total_steps={args.num_training_steps}"
        )
        print(f"Training hyperparameters={args.training_hparams_source}")

    for epoch in range(start_epoch, args.epochs):
        train_sampler = getattr(loaders["train"], "sampler", None)
        if distributed and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        tic = time.time()
        train_loss, train_acc = run_epoch(
            loaders["train"],
            model,
            criterion,
            optimizer,
            scheduler,
            scheduler_interval,
            scaler,
            device,
            train=True,
            print_freq=args.print_freq,
            epoch=epoch + 1,
        )
        val_loss, val_acc = run_epoch(
            loaders["dev"],
            model,
            criterion,
            optimizer=None,
            scheduler=None,
            scheduler_interval=scheduler_interval,
            scaler=None,
            device=device,
            train=False,
            print_freq=args.print_freq,
            epoch=epoch + 1,
        )
        if scheduler is not None and scheduler_interval == "epoch":
            scheduler.step()

        test_loss = test_acc = None
        if args.eval_test_every_epoch:
            if data.validation_uses_test:
                test_loss, test_acc = val_loss, val_acc
            else:
                test_loss, test_acc = run_epoch(
                    loaders["test"],
                    model,
                    criterion,
                    optimizer=None,
                    scheduler=None,
                    scheduler_interval=scheduler_interval,
                    scaler=None,
                    device=device,
                    train=False,
                    print_freq=args.print_freq,
                    epoch=epoch + 1,
                )

        is_best = val_acc > best_val
        best_val = max(best_val, val_acc)
        state = {
            "epoch": epoch + 1,
            "best_val": best_val,
            "state_dict": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "args": vars(args),
            "task_spec": spec.__dict__,
        }
        if is_main_process():
            save_checkpoint(state, output_dir, is_best=is_best)

        lr = optimizer.param_groups[0]["lr"]
        msg = (
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f} "
            f"best_val={best_val:.2f} lr={lr:.6g} time={time.time() - tic:.1f}s"
        )
        if test_acc is not None:
            msg += f" test_loss={test_loss:.4f} test_acc={test_acc:.2f}"
        if is_main_process():
            print(msg, flush=True)

    if distributed:
        dist.barrier()
    best_path = output_dir / "model_best.pth.tar"
    if best_path.exists():
        best = torch.load(best_path, map_location=device,weights_only=False)
        unwrap_model(model).load_state_dict(best["state_dict"])
    test_loss, test_acc = run_epoch(
        loaders["test"],
        model,
        criterion,
        optimizer=None,
        scheduler=None,
        scheduler_interval=scheduler_interval,
        scaler=None,
        device=device,
        train=False,
        print_freq=args.print_freq,
        epoch=args.epochs,
    )
    if is_main_process():
        print(f"Best checkpoint test_loss={test_loss:.4f} test_acc={test_acc:.2f}")
    if distributed_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
