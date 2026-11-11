"""Spiking statistics of a trained NR-DN network (rebuttal R4-Q5, Table R4-5).

At inference, an integer soma output S_t in {L, ..., U} is emitted as |S_t| unit
spikes of sign sgn(S_t) over D = max(|L|, U) virtual steps (MMDEND / Luo et al.),
so W S_t = sum_u W S_{t,u} and a layer fed by spikes only accumulates (ACs).
All numbers are per test sample:

* firing rate: mean |S_t| over neurons and time steps, i.e. unit spikes per
  neuron per step (also reported per virtual step, mean |S_t| / D);
* SOPs: accumulations of every weight layer fed by spikes, i.e. the sum over
  unit spikes of their fan-out. It is counted exactly, including the zero
  padding of convolutions. An average pool between a soma and the next layer
  is folded into that layer (pooling is linear), so each spike reaches the
  fan-out of its pooled position;
* MACs: (i) weight layers whose input is real-valued (the image input of
  S-CIFAR) and (ii) the neurons, counted with the per-step FLOPs of Table 3
  (dendrite 3dK + d(9K - 3) + 3, soma 2W) and charged as one MAC each, as in
  the energy formula of MMDEND;
* energy = E_MAC * MACs + E_AC * SOPs with E_AC = 0.9 pJ and E_MAC = 4.6 pJ (45 nm).

BatchNorm right after a weight layer is folded into it at inference and Dropout
is the identity, so neither is counted; biases are not counted. The network input
is treated as spikes (event counts) when it is integer-valued.
"""
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from distal_stimulation import SOMA_TYPES
from module.dend_compartment import SparseChannelPreservingTrunkDistalDendCompartment

E_AC_PJ = 0.9
E_MAC_PJ = 4.6
WEIGHT_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d)
POOL_TYPES = (nn.AvgPool1d, nn.AvgPool2d)
RESHAPE_TYPES = (nn.Flatten,)
DEND_TYPE = SparseChannelPreservingTrunkDistalDendCompartment


def dendrite_flops(dend):
    """Table 3, per neuron and step: compartment update 3dK + DMPR readout d(9K - 3) + 3."""
    if dend.branch_readout_mode != "trunk_distal":
        raise NotImplementedError("Table 3 gives the cost of the DMPR readout only")
    d, K = dend.branch_degree, dend.compartments_per_branch
    return 3 * d * K + d * (9 * K - 3) + 3


def soma_flops(soma, T):
    """Table 3, per neuron and step: 2W for a causal window of length W (at most T)."""
    order = getattr(soma, "psn_order", getattr(soma, "order", None))
    if order is None:
        raise NotImplementedError(f"no Table 3 cost for soma {type(soma).__name__}")
    return 2 * min(int(order), T)


def soma_levels(soma):
    """D = max(|L|, U): unit spikes needed for the largest integer output."""
    return max(abs(int(getattr(soma, "L", 0))), int(getattr(soma, "thre", 1)))


def _pool_size(pool):
    k = pool.kernel_size
    if isinstance(k, int):
        k = (k,) * (1 if isinstance(pool, nn.AvgPool1d) else 2)
    return math.prod(k)


def accumulations(layer, counts):
    """Exact number of ACs when ``counts`` unit spikes ([T, N, ...], >= 0) enter ``layer``.

    It equals the sum of the layer's output with all weights set to one and no
    bias: each input spike adds one weight to every output that it reaches.
    """
    if isinstance(layer, nn.Linear):
        return counts.sum(dtype=torch.float64).item() * layer.out_features
    if layer.groups != 1 or layer.padding_mode != "zeros":
        raise NotImplementedError("only zero-padded convolutions with groups=1")
    conv = F.conv1d if isinstance(layer, nn.Conv1d) else F.conv2d
    x = counts.flatten(0, 1)                                    # [T * N, C_in, ...]
    ones = torch.ones(1, layer.in_channels, *layer.kernel_size, dtype=x.dtype, device=x.device)
    reach = conv(x, ones, None, layer.stride, layer.padding, layer.dilation)
    return reach.sum(dtype=torch.float64).item() * layer.out_channels


def dense_macs(layer, out):
    """MACs of ``layer`` on a real-valued input: one MAC per weight of every output."""
    return float(out.numel()) * layer.weight[0].numel()


class _Counter:
    """Forward hooks that follow the unit-spike counts from each soma to the next weight layer."""

    def __init__(self, model):
        self.names = {module: name for name, module in model.named_modules()}
        self.layers = {}            # name -> totals, in forward order
        self.spikes = None          # unit-spike counts on their way to the next weight layer
        self.busy = False           # set while a hook re-applies a module to the counts
        self.hooks = [
            module.register_forward_hook(self.hook)
            for module in self.names
            if isinstance(module, SOMA_TYPES + WEIGHT_TYPES + POOL_TYPES + RESHAPE_TYPES + (DEND_TYPE,))
        ]

    def remove(self):
        for hook in self.hooks:
            hook.remove()

    def _entry(self, module, kind, **init):
        name = self.names[module]
        if name not in self.layers:
            self.layers[name] = {"layer": name, "kind": kind, **init}
        return self.layers[name]

    def _apply(self, module, x):
        self.busy = True
        try:
            return module(x)
        finally:
            self.busy = False

    @torch.no_grad()
    def hook(self, module, args, out):
        if self.busy:
            return
        x = args[0]
        if isinstance(module, SOMA_TYPES):
            if not torch.equal(out, out.round()):
                raise RuntimeError(f"{self.names[module]}: soma output is not integer-valued")
            counts = out.abs()
            e = self._entry(module, "soma", D=soma_levels(module), unit_spikes=0.0,
                            silent=0.0, neuron_steps=0.0, neuron_flops=0.0)
            e["unit_spikes"] += counts.sum(dtype=torch.float64).item()
            e["silent"] += (out == 0).sum().item()
            e["neuron_steps"] += out.numel()
            e["neuron_flops"] += float(out.numel()) * soma_flops(module, out.shape[0])
            self.spikes = counts
        elif isinstance(module, DEND_TYPE):
            e = self._entry(module, "dendrite", neuron_flops=0.0)
            e["neuron_flops"] += float(out.numel()) * dendrite_flops(module)
            self.spikes = None
        elif isinstance(module, POOL_TYPES):
            if self.spikes is not None:          # sum pooling of the counts = folded average pool
                self.spikes = self._apply(module, self.spikes) * _pool_size(module)
        elif isinstance(module, RESHAPE_TYPES):
            if self.spikes is not None:
                self.spikes = self._apply(module, self.spikes)
        else:                                     # weight layer
            counts, source = self.spikes, "spikes"
            if counts is None and torch.equal(x, x.round()):
                counts, source = x.abs(), "integer input"
            if counts is not None and counts.shape != x.shape:
                raise RuntimeError(f"{self.names[module]}: spike counts {tuple(counts.shape)} "
                                   f"do not match the layer input {tuple(x.shape)}")
            e = self._entry(module, "weight", input="dense" if counts is None else source,
                            sops=0.0, macs=0.0, dense_equivalent_macs=0.0)
            if counts is None:
                e["macs"] += dense_macs(module, out)
            else:
                e["sops"] += accumulations(module, counts)
            e["dense_equivalent_macs"] += dense_macs(module, out)
            self.spikes = None


@torch.no_grad()
def run_spike_statistics(model, make_batches, max_samples=None):
    """Table R4-5 statistics on the test set (``max_samples=None``: all of it)."""
    model.eval()
    counter = _Counter(model)
    n = 0
    try:
        for inputs in make_batches():
            if max_samples is not None:
                if n >= max_samples:
                    break
                inputs = inputs[: max_samples - n]
            counter.spikes = None
            functional.reset_net(model)
            model(inputs)
            functional.reset_net(model)
            n += inputs.shape[0]
    finally:
        counter.remove()
    if n == 0:
        raise RuntimeError("no test samples were analysed")

    layers = list(counter.layers.values())
    somas = [l for l in layers if l["kind"] == "soma"]
    weights = [l for l in layers if l["kind"] == "weight"]
    if not somas:
        raise RuntimeError("no soma found in the model")
    for l in somas:
        l["firing_rate"] = l["unit_spikes"] / l["neuron_steps"]
        l["firing_rate_per_virtual_step"] = l["firing_rate"] / l["D"]
        l["silent_share"] = l["silent"] / l["neuron_steps"]

    def total(kind, key):
        return sum(l[key] for l in layers if l["kind"] == kind)

    neuron_steps = total("soma", "neuron_steps")
    sops = total("weight", "sops") / n
    macs_input = total("weight", "macs") / n
    macs_dendrite = total("dendrite", "neuron_flops") / n
    macs_soma = total("soma", "neuron_flops") / n
    macs = macs_input + macs_dendrite + macs_soma
    per_sample = {
        "firing_rate": total("soma", "unit_spikes") / neuron_steps,
        "firing_rate_per_virtual_step": sum(l["unit_spikes"] / l["D"] for l in somas) / neuron_steps,
        "silent_share": total("soma", "silent") / neuron_steps,
        "sops": sops,
        "macs": macs,
        "macs_dense_layers": macs_input,
        "macs_dendrite": macs_dendrite,
        "macs_soma": macs_soma,
        "energy_uj": (E_MAC_PJ * macs + E_AC_PJ * sops) * 1e-6,
        # same network with real-valued activations: every weight layer as MACs
        "dense_equivalent_synaptic_macs": total("weight", "dense_equivalent_macs") / n,
    }
    return {
        "settings": {"samples": n, "E_AC_pJ": E_AC_PJ, "E_MAC_pJ": E_MAC_PJ,
                     "neuron_cost": "Table 3 FLOPs per neuron and step, each charged as one MAC"},
        "per_sample": per_sample,
        "layers": layers,
        "weight_layer_inputs": {l["layer"]: l["input"] for l in weights},
    }


def save_and_print(result, path):
    with open(path, "w") as fid:
        json.dump(result, fid, indent=2)
    s = result["per_sample"]
    mega = lambda v: f"{v / 1e6:.3f} M"
    print(f"spike statistics: {result['settings']['samples']} test samples, saved to {path}")
    print("Table R4-5 cells (per test sample):")
    print(f"  firing rate (unit spikes / neuron / step): {s['firing_rate']:.4f}"
          f"   (per virtual step {s['firing_rate_per_virtual_step']:.4f}, silent {s['silent_share']:.1%})")
    print(f"  SOPs (ACs)                               : {mega(s['sops'])}")
    print(f"  MACs                                     : {mega(s['macs'])}"
          f"   (dense layers {mega(s['macs_dense_layers'])}, dendrite {mega(s['macs_dendrite'])}, "
          f"soma {mega(s['macs_soma'])})")
    print(f"  estimated energy (uJ)                    : {s['energy_uj']:.3f}")
    print(f"  reference: synaptic MACs with real-valued activations = {mega(s['dense_equivalent_synaptic_macs'])}")
    for l in result["layers"]:
        if l["kind"] == "soma":
            print(f"    {l['layer']}: D {l['D']}, rate {l['firing_rate']:.4f}, silent {l['silent_share']:.1%}")
        elif l["kind"] == "weight":
            n = result["settings"]["samples"]
            print(f"    {l['layer']}: input {l['input']}, SOPs {mega(l['sops'] / n)}, MACs {mega(l['macs'] / n)}")