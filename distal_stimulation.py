"""Analyses of trained NR-DN layers for the rebuttal.

* ``run_dmpr_statistics`` (R3-Q2, Table R3-2): learned gamma / rho and the size of
  the DMPR modulation m on test data, per layer.
* ``run_distal_stimulation`` (R2-W1, Table R2-1): in-silico distal stimulation.

In-silico distal stimulation:

An in-silico counterpart of the paired-stimulation experiment of Behabadi et al.
(2012), where distal excitation lowers the threshold of proximally driven
responses. For every NR-DN layer (dendrite + soma) of a trained network:

1. the layer input X is recorded on test samples and kept fixed;
2. for a random subset (``fraction``) of its neurons (channels), the distal
   compartment states are shifted, z_k <- z_k + Delta * sigma_k (k >= 1), where
   sigma_k is the standard deviation of z_k on the analysed samples, while the
   proximal state is scaled, z_0 <- kappa * z_0 (kappa = 1: unchanged);
3. the branch readout, Eq. 11 and the soma are recomputed from these states.

Only a small subset is stimulated because delta-hat is RMS-normalized across
channels: shifting every neuron would largely be normalized away.

Metrics, computed on the stimulated neurons only:
* firing rate r(kappa, Delta) = mean |S_t| = spikes per neuron per time step
  after the integer-to-spike conversion;
* proximal threshold kappa*(Delta) = smallest kappa with r(kappa, Delta) >= r(1, 0)
  (linear interpolation on the kappa grid), normalized by kappa*(0) so that
  kappa*(0) = 1. kappa* < 1: a weaker proximal response suffices.
Both are averaged over the random subsets and over all NR-DN layers.
"""
import json
import math

import torch
from spikingjelly.activation_based import functional

from module import soma as soma_module
from module.dend_compartment import SparseChannelPreservingTrunkDistalDendCompartment

DEFAULT_DELTAS = (-1.0, 0.0, 1.0, 2.0)
DEFAULT_KAPPAS = tuple(round(0.2 + 0.05 * i, 2) for i in range(37))   # 0.20 ... 2.00
SOMA_TYPES = (
    soma_module.PSNIntergerSoma_ssf,
    soma_module.ExponentialSoma_ssf,
    soma_module.MaskedSlidingPSN,
)


def find_dend_soma_pairs(model):
    """(name, dendrite, soma) of every NR-DN layer, in forward order."""
    pairs, pending = [], None
    for name, module in model.named_modules():
        if isinstance(module, SparseChannelPreservingTrunkDistalDendCompartment):
            pending = (name, module)
        elif pending is not None and isinstance(module, SOMA_TYPES):
            pairs.append((pending[0], pending[1], module))
            pending = None
    if not pairs:
        raise RuntimeError("no dendrite + soma layer found in the model")
    return pairs


def _layer_inputs(model, pairs, make_batches, max_samples):
    """Yield, per test batch, the recorded input of every NR-DN layer."""
    captured = {}
    hooks = []
    for index, (_, dend, _) in enumerate(pairs):
        def hook(module, args, index=index):
            captured[index] = args[0].detach()
        hooks.append(dend.register_forward_pre_hook(hook))
    try:
        seen = 0
        for inputs in make_batches():
            if max_samples is not None:
                if seen >= max_samples:
                    break
                inputs = inputs[: max_samples - seen]
            captured.clear()
            functional.reset_net(model)
            model(inputs)
            functional.reset_net(model)
            layer_inputs = [captured[i] for i in range(len(pairs))]
            seen += inputs.shape[0]
            yield layer_inputs
    finally:
        for hook in hooks:
            hook.remove()


def _states(dend, x):
    """Compartment states [T, N, d, C, K, *spatial] of a dendrite for input x."""
    dend.reset()
    previous = dend.store_v_seq
    dend.store_v_seq = True
    dend(x)
    states = dend.v_seq
    dend.store_v_seq = previous
    dend.reset()
    return states


def _modulation(dend, states, x):
    """DMPR modulation m = tanh(sum_k pi_k p_k tanh(delta_hat_k)), [T, N, d, C, *spatial]."""
    previous = dend.store_branch_monitor
    dend.store_branch_monitor = True
    dend._compute_branch_output_sequence(states, x)
    modulation = dend.branch_mod_seq
    dend.store_branch_monitor = previous
    dend.branch_mod_seq = dend.distal_gate_seq = dend.branch_output_seq = None
    return modulation


@torch.no_grad()
def run_dmpr_statistics(model, make_batches, max_samples=None, threshold=0.05):
    """Rebuttal R3-Q2 (Table R3-2): DMPR statistics of every NR-DN layer.

    Per layer: gamma_j and |rho_j| averaged over the B branch templates, the
    share of templates with gamma_j > 0, and, over all test samples, time steps
    and neurons, the mean |m|, the mean |gamma_j m| (relative change of the
    proximal response by the multiplicative modulation) and the share of entries
    with |gamma_j m| > ``threshold``. ``max_samples=None`` uses the whole test set.
    """
    model.eval()
    pairs = [p for p in find_dend_soma_pairs(model) if p[1].branch_readout_mode == "trunk_distal"]
    if not pairs:
        raise RuntimeError("no DMPR (trunk_distal) layer found in the model")
    totals = [{"m": 0.0, "gm": 0.0, "over": 0.0, "flip": 0.0, "n": 0} for _ in pairs]
    for layer_inputs in _layer_inputs(model, pairs, make_batches, max_samples):
        for i, (_, dend, _) in enumerate(pairs):
            x = layer_inputs[i]
            m = _modulation(dend, _states(dend, x), x)
            gamma = dend._edge_branch_value(dend.distal_gain.detach().to(dtype=m.dtype, device=m.device))
            gamma = gamma.view(1, 1, dend.branch_degree, dend.channels, *([1] * (m.dim() - 4)))
            signed_gm = gamma * m
            gm = signed_gm.abs()
            totals[i]["flip"] += (signed_gm < -1.0).sum().double().item()   # gain 1 + gamma m < 0
            totals[i]["m"] += m.abs().sum().double().item()
            totals[i]["gm"] += gm.sum().double().item()
            totals[i]["over"] += (gm > threshold).sum().double().item()
            totals[i]["n"] += m.numel()

    layers = []
    all_gamma, all_rho = [], []
    for (name, dend, _), total in zip(pairs, totals):
        gamma = dend.distal_gain.detach().float().cpu()
        rho = dend.distal_residual_gain.detach().float().cpu()
        all_gamma.append(gamma)
        all_rho.append(rho)
        layers.append({
            "layer": name,
            "gamma_mean": gamma.mean().item(),
            "gamma_min": gamma.min().item(),
            "gamma_max": gamma.max().item(),
            "gamma_positive_share": (gamma > 0).float().mean().item(),
            "abs_gamma_mean": gamma.abs().mean().item(),
            "abs_m_mean": total["m"] / total["n"],
            "abs_gamma_m_mean": total["gm"] / total["n"],
            "abs_gamma_m_over_threshold": total["over"] / total["n"],
            "abs_rho_mean": rho.abs().mean().item(),
            "rho_mean": rho.mean().item(),
            "rho_positive_share": (rho > 0).float().mean().item(),
            "gain_sign_flip_share": total["flip"] / total["n"],
        })

    def across_layers(key):
        values = [layer[key] for layer in layers]
        return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}

    return {
        "threshold": threshold,
        "max_samples": max_samples,
        "summary": {
            "gamma": across_layers("gamma_mean"),
            "gamma_positive_share": (torch.cat(all_gamma) > 0).float().mean().item(),
            "abs_gamma": across_layers("abs_gamma_mean"),
            "rho_positive_share": (torch.cat(all_rho) > 0).float().mean().item(),
            "gain_sign_flip": across_layers("gain_sign_flip_share"),
            "abs_m": across_layers("abs_m_mean"),
            "abs_gamma_m": across_layers("abs_gamma_m_mean"),
            "abs_gamma_m_over_threshold": across_layers("abs_gamma_m_over_threshold"),
            "abs_rho": across_layers("abs_rho_mean"),
        },
        "layers": layers,
    }


def save_and_print_stats(result, path):
    """Save Table R3-2 statistics and print one ready-to-paste cell per row."""
    with open(path, "w") as fid:
        json.dump(result, fid, indent=2)
    summary = result["summary"]

    def cell(entry, percent=False):
        if percent:
            return f"{entry['mean']:.1%} ({entry['min']:.1%}–{entry['max']:.1%})"
        return f"{entry['mean']:.3f} ({entry['min']:.3f}–{entry['max']:.3f})"

    print(f"DMPR statistics: {len(result['layers'])} NR-DN layers, saved to {path}")
    print("cells: mean over layers (min–max across layers)")
    print(f"  |gamma_j|; share gamma > 0 : {cell(summary['abs_gamma'])}; {summary['gamma_positive_share']:.0%}")
    print(f"  gamma_j (signed)           : {cell(summary['gamma'])}")
    print(f"  |m|                        : {cell(summary['abs_m'])}")
    print(f"  |gamma_j m|                : {cell(summary['abs_gamma_m'])}")
    print(f"  share |gamma_j m| > {result['threshold']:<6}: {cell(summary['abs_gamma_m_over_threshold'], percent=True)}")
    print(f"  |rho_j|; share rho > 0     : {cell(summary['abs_rho'])}; {summary['rho_positive_share']:.0%}")
    print(f"  share gain 1+gamma m < 0   : {cell(summary['gain_sign_flip'], percent=True)}")
    for layer in result["layers"]:
        print(f"    {layer['layer']}: gamma {layer['gamma_mean']:.3f} [{layer['gamma_min']:.3f}, {layer['gamma_max']:.3f}], "
              f"|gamma| {layer['abs_gamma_mean']:.3f}, rho {layer['rho_mean']:.3f} (>0: {layer['rho_positive_share']:.0%}), "
              f"gain<0: {layer['gain_sign_flip_share']:.1%}, "
              f"|m| {layer['abs_m_mean']:.3f}, |gamma m| {layer['abs_gamma_m_mean']:.4f}, "
              f">{result['threshold']}: {layer['abs_gamma_m_over_threshold']:.1%}, |rho| {layer['abs_rho_mean']:.3f}")


def _fire(dend, soma, states, x):
    """Branch readout + Eq. 11 + soma from (possibly modified) states."""
    edge_output = dend._compute_branch_output_sequence(states, x)
    somatic_input = dend._merge_edges_sequence(edge_output, x)
    soma.reset()
    spikes = soma(somatic_input)
    soma.reset()
    return spikes


def _threshold(kappas, rates, target):
    """Smallest kappa with rate >= target (linear interpolation); nan if never."""
    for b, rate in enumerate(rates):
        if rate >= target:
            if b == 0:
                return kappas[0]
            k0, k1, r0, r1 = kappas[b - 1], kappas[b], rates[b - 1], rates[b]
            return k0 + (target - r0) * (k1 - k0) / (r1 - r0)
    return float("nan")


def _nanmean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")


@torch.no_grad()
def run_distal_stimulation(
    model,
    make_batches,
    max_samples=1024,
    deltas=DEFAULT_DELTAS,
    kappas=DEFAULT_KAPPAS,
    fraction=0.1,
    n_subsets=5,
    seed=0,
):
    """Run the experiment. ``make_batches()`` must yield model inputs on the device."""
    deltas = [float(d) for d in deltas]
    kappas = sorted(float(k) for k in kappas)
    if 0.0 not in deltas:
        deltas = sorted(deltas + [0.0])
    if 1.0 not in kappas:
        kappas = sorted(kappas + [1.0])
    i_delta0, i_kappa1 = deltas.index(0.0), kappas.index(1.0)

    model.eval()
    pairs = find_dend_soma_pairs(model)
    n_layers = len(pairs)

    generator = torch.Generator().manual_seed(seed)
    masks = []
    for _, dend, _ in pairs:
        n_selected = max(1, round(fraction * dend.channels))
        layer_masks = []
        for _ in range(n_subsets):
            mask = torch.zeros(dend.channels, dtype=torch.bool)
            mask[torch.randperm(dend.channels, generator=generator)[:n_selected]] = True
            layer_masks.append(mask)
        masks.append(layer_masks)

    # pass 1: sigma_k of every compartment, per layer
    sums = [0.0] * n_layers
    squares = [0.0] * n_layers
    counts = [0] * n_layers
    for layer_inputs in _layer_inputs(model, pairs, make_batches, max_samples):
        for i, (_, dend, _) in enumerate(pairs):
            z = _states(dend, layer_inputs[i]).movedim(4, 0)
            z = z.reshape(z.shape[0], -1).double()
            sums[i] = sums[i] + z.sum(dim=1)
            squares[i] = squares[i] + z.square().sum(dim=1)
            counts[i] += z.shape[1]
    sigmas = [
        (squares[i] / counts[i] - (sums[i] / counts[i]).square()).clamp_min(0).sqrt().float()
        for i in range(n_layers)
    ]

    # pass 2: firing of the stimulated neurons for every (Delta, kappa)
    spike_sums = torch.zeros(n_layers, n_subsets, len(deltas), len(kappas), dtype=torch.float64)
    spike_counts = torch.zeros(n_layers, n_subsets, dtype=torch.float64)
    for layer_inputs in _layer_inputs(model, pairs, make_batches, max_samples):
        for i, (_, dend, soma) in enumerate(pairs):
            x = layer_inputs[i]
            z = _states(dend, x)
            spatial = [1] * (z.dim() - 5)
            K = z.shape[4]
            sigma = sigmas[i].to(device=z.device, dtype=z.dtype).clone()
            sigma[0] = 0.0                                              # proximal is not shifted
            sigma = sigma.view(1, 1, 1, 1, K, *spatial)
            proximal = torch.zeros(K, device=z.device, dtype=z.dtype)
            proximal[0] = 1.0
            proximal = proximal.view(1, 1, 1, 1, K, *spatial)
            for s, mask in enumerate(masks[i]):
                mask = mask.to(z.device)
                state_mask = mask.to(z.dtype).view(1, 1, 1, -1, 1, *spatial)
                out_mask = mask.view(1, 1, -1, *spatial)
                spike_counts[i, s] += float(out_mask.expand(x.shape).sum().item())
                for a, delta in enumerate(deltas):
                    z_delta = z + state_mask * (delta * sigma)
                    for b, kappa in enumerate(kappas):
                        z_mod = z_delta * (1.0 + state_mask * (kappa - 1.0) * proximal)
                        spikes = _fire(dend, soma, z_mod, x)
                        spike_sums[i, s, a, b] += (spikes.abs() * out_mask).sum().double().item()
    rates = spike_sums / spike_counts[:, :, None, None]

    layers = []
    for i, (name, dend, _) in enumerate(pairs):
        firing, thresholds = [], []
        for a in range(len(deltas)):
            firing.append(rates[i, :, a, i_kappa1].mean().item())
            per_subset = []
            for s in range(n_subsets):
                target = rates[i, s, i_delta0, i_kappa1].item()
                k_delta = _threshold(kappas, rates[i, s, a].tolist(), target)
                k_zero = _threshold(kappas, rates[i, s, i_delta0].tolist(), target)
                per_subset.append(k_delta / k_zero if k_zero > 0 else float("nan"))
            thresholds.append(_nanmean(per_subset))
        entry = {"layer": name, "readout": dend.branch_readout_mode,
                 "firing_rate": firing, "kappa_star": thresholds}
        if dend.branch_readout_mode == "trunk_distal":
            gamma = dend.distal_gain.detach().float().cpu()
            rho = dend.distal_residual_gain.detach().float().cpu()
            entry.update(gamma=gamma.tolist(), rho=rho.tolist(),
                         gamma_positive_share=(gamma > 0).float().mean().item())
        layers.append(entry)

    return {
        "deltas": deltas,
        "kappas": kappas,
        "settings": {"max_samples": max_samples, "fraction": fraction,
                     "n_subsets": n_subsets, "seed": seed},
        "mean": {
            "firing_rate": [_nanmean([l["firing_rate"][a] for l in layers]) for a in range(len(deltas))],
            "kappa_star": [_nanmean([l["kappa_star"][a] for l in layers]) for a in range(len(deltas))],
        },
        "layers": layers,
    }


def save_and_print(result, path):
    with open(path, "w") as fid:
        json.dump(result, fid, indent=2)
    print(f"distal stimulation: {len(result['layers'])} NR-DN layers, saved to {path}")
    print(f"{'Delta':>6} | {'firing rate':>11} | {'kappa*':>7}   (mean over layers and subsets)")
    for a, delta in enumerate(result["deltas"]):
        print(f"{delta:>6.2f} | {result['mean']['firing_rate'][a]:>11.4f} | {result['mean']['kappa_star'][a]:>7.3f}")
    for layer in result["layers"]:
        if "gamma" in layer:
            print(f"  {layer['layer']}: share of templates with gamma > 0 = {layer['gamma_positive_share']:.0%}")