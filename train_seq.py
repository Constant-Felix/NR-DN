import math
import torch
import torch.nn as nn
import torch.nn.functional as F
_seed_ = 2020
import random
random.seed(2020)
import numpy as np
import torchvision
from torchvision import transforms
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.activation_based import functional, surrogate, layer
from spikingjelly.activation_based import neuron as ner
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
from module import dendrite,dend_compartment,soma,neuron,wiring
from module.dend_compartment import SparseChannelPreservingTrunkDistalDendCompartment



class ClassificationPresetTrain:
    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = []
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

from torch import Tensor
from typing import Tuple
class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s
class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = torchvision.transforms.functional.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

# Rebuttal W4 (Tables R1-4 / R1-5): somatic kernel and firing range [L, U].
# "free": learnable sliding window of length `order` (default NR-DN soma);
# "exp":  exponential kernel over all T lags (MMDEND-style), same firing rule.
SOMA_KERNELS = ("free", "exp")


def make_soma(kernel, order, L, thre):
    if kernel == "exp":
        return soma.ExponentialSoma_ssf(L=L, thre=thre)
    return soma.PSNIntergerSoma_ssf(psn_order=order, L=L, thre=thre, psn_exp_init=True, psn_backend='fft')


# Rebuttal W5 (Tables R1-6 / R1-7): proximal role and intra-branch coupling.
# Within a branch tau_k = tau_branch * scale_k and compartment 0 is the proximal
# one of DMPR, so "slowest" makes the slowest compartment proximal.
PROXIMAL_TAU_SCALE = {"fastest": (1.0, 2.5), "slowest": (2.5, 1.0)}
COUPLINGS = ("none", "chain")


def dend_class(coupling):
    if coupling == "chain":
        return dend_compartment.ChainCoupledSparseDendCompartment
    return SparseChannelPreservingTrunkDistalDendCompartment


class CIFAR10Net(nn.Module):
    def __init__(self, channels, class_num: int, T: int=32, P:int=-1,L=-4,thre=4,num_branches=8,compartments_per_branch=4,branch_degree=1,readout='trunk_distal',readout_act='gelu',soma_kernel='free',coupling='none',proximal='fastest',dmpr_ablation='none'):
        super().__init__()
        dend_cls = dend_class(coupling)
        tau_scale = PROXIMAL_TAU_SCALE[proximal]
        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels
                conv.append(layer.Conv1d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm1d(channels))
                conv.append(dend_cls(channels,num_branches=num_branches,compartments_per_branch=compartments_per_branch,c_sub=channels,branch_degree=branch_degree,learn_comp_gain=True,learn_edge_gain=True,integration_backend='fft',merge_norm='mean',branch_readout_mode=readout,readout_activation=readout_act,compartment_tau_scale=tau_scale,dmpr_ablation=dmpr_ablation))
                conv.append(make_soma(soma_kernel, T//8, L, thre))
                #conv.append(soma.MaskedSlidingPSN(order=T,exp_init=True,backend='fft'))

            conv.append(layer.AvgPool1d(2))

        self.conv = nn.Sequential(*conv)


        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(channels * 8, channels * 8 // 4),
            dend_cls(channels * 2,num_branches=num_branches,compartments_per_branch=compartments_per_branch,c_sub=channels,branch_degree=branch_degree,learn_comp_gain=True,learn_edge_gain=True,integration_backend='fft',merge_norm='mean',branch_readout_mode=readout,readout_activation=readout_act,compartment_tau_scale=tau_scale,dmpr_ablation=dmpr_ablation),
            #soma.MaskedSlidingPSN(order=T,exp_init=True,backend='fft'),
            make_soma(soma_kernel, T//8, L, thre),
            layer.Linear(channels * 8 // 4, class_num),
        )

        functional.set_step_mode(self, 'm')

    def forward(self, x_seq: torch.Tensor):
        # [N, C, H, W] -> [W, N, C, H]
        x_seq = x_seq.permute(3, 0, 1, 2)
        x_seq = self.fc(self.conv(x_seq))  # [W, N, C]
        return x_seq.mean(0)

# python train_seq.py -data-dir /data2/hyx/ViT-dend/data/cifar100 -amp -class-num 100 -channels 128  -warmup-epochs 0  -epochs 400 -opt adamw -lr 0.001 -dmpr-stats exp/pt/None_e400_b128_adamw_lr0.001_c128_20261110-045305_amp_coup-chain_P32/checkpoint_max.pth -coupling chain   -resume logs/pt/None_e400_b128_adamw_lr0.001_c128_20261016-060357_amp_P32/checkpoint_latest.pth

from datetime import datetime
def main():

    parser = argparse.ArgumentParser(description='Classify Sequential CIFAR10/100')
    parser.add_argument('-device', default='cuda:7', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=256, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of CIFAR10/100 dataset')
    parser.add_argument('-out-dir', type=str, default='/data3/hyx/NR-DN/exp/', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-wd', default=0.0, type=float, metavar='W', help='weight decay')
    parser.add_argument('-min-lr', '--min-lr', default=0, type=float,
                        help='minimum learning rate for cosine scheduler')
    parser.add_argument('-warmup-epochs', '--warmup-epochs', default=20, type=int,
                        help='number of warmup epochs before cosine decay')
    parser.add_argument('-warmup-lr', '--warmup-lr', default=1e-4, type=float,
                        help='initial learning rate used at the start of warmup')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-neu', type=str, help='use which neuron')
    parser.add_argument('-class-num', type=int, default=10)

    parser.add_argument('-P', default=32, type=int, help='the order of the masked/sliding PSN')
    parser.add_argument('-exp-init', action='store_true', help='use the exp init method to initialize the weight of SPSN')
    # Rebuttal W1 (Table R1-1): branch readout; trunk_distal = DMPR (default)
    parser.add_argument('-readout', default='trunk_distal', choices=list(dend_compartment.BRANCH_READOUT_MODES), help='branch readout of the dendrite')
    parser.add_argument('-readout-act', default='gelu', choices=list(dend_compartment.READOUT_ACTIVATIONS), help='activation of -readout linear_act')
    # Rebuttal W4 (Tables R1-4 / R1-5): firing range and somatic kernel
    parser.add_argument('-L', default=-4, type=int, help='lower firing bound L')
    parser.add_argument('-U', default=4, type=int, help='upper firing bound U')
    parser.add_argument('-soma-kernel', default='free', choices=list(SOMA_KERNELS), help='free sliding window (W=T//8) or exponential kernel')
    # Rebuttal W5 (Tables R1-6 / R1-7): proximal compartment and intra-branch coupling
    parser.add_argument('-proximal', default='fastest', choices=list(PROXIMAL_TAU_SCALE), help='which compartment (by initial tau) is proximal in each branch')
    parser.add_argument('-coupling', default='none', choices=list(COUPLINGS), help='none: diagonal A; chain: symmetric tridiagonal A')
    # Rebuttal R4-Q4 (Table R4-3): remove one DMPR component (-readout trunk_distal only)
    parser.add_argument('-dmpr-ablation', default='none', choices=list(dend_compartment.DMPR_ABLATIONS), help='no_gate: p_k = 1; no_tanh: delta_hat instead of tanh(delta_hat)')
    # Rebuttal analyses of a trained model (no training): R3-Q2 DMPR statistics, R2-W1 distal stimulation
    parser.add_argument('-dmpr-stats', type=str, default='', help='checkpoint (checkpoint_max.pth): learned gamma/rho and |m| statistics (Table R3-2)')
    parser.add_argument('-distal-stim', type=str, default='', help='checkpoint (checkpoint_max.pth): in-silico distal stimulation (Table R2-1)')
    parser.add_argument('-stim-samples', default=1024, type=int, help='number of test samples used by -distal-stim')
    # Rebuttal R4-Q5 (Table R4-5): firing rate, SOPs, MACs and energy of a trained model
    parser.add_argument('-spike-stats', type=str, default='', help='checkpoint (checkpoint_max.pth): spiking statistics (Table R4-5)')
    parser.add_argument('-spike-samples', default=None, type=int, help='number of test samples used by -spike-stats (default: whole test set)')
    parser.add_argument('-stim-fraction', default=0.1, type=float, help='share of neurons stimulated per subset')
    parser.add_argument('-stim-subsets', default=5, type=int, help='number of random neuron subsets per layer')


    args = parser.parse_args()
    print(args)

    mixup_transforms = []
    mixup_transforms.append(RandomMixup(args.class_num, p=1.0, alpha=0.2))
    mixup_transforms.append(RandomCutmix(args.class_num, p=1.0, alpha=1.))
    mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

    if args.class_num == 10:
        transform_train = ClassificationPresetTrain(mean=(0.4914, 0.4822, 0.4465),
                                                      std=(0.2023, 0.1994, 0.2010), interpolation=InterpolationMode('bilinear'),
                                                      auto_augment_policy='ta_wide',
                                                      random_erase_prob=0.1)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.class_num == 100:
        transform_train = ClassificationPresetTrain(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                    std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                                                    interpolation=InterpolationMode('bilinear'),
                                                    auto_augment_policy='ta_wide',
                                                    random_erase_prob=0.1)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])

    else:
        raise NotImplementedError(args.class_num)



    if args.class_num == 10:
        train_set = torchvision.datasets.CIFAR10(
                root=args.data_dir,
                train=True,
                transform=transform_train,
                download=True)

        test_set = torchvision.datasets.CIFAR10(
                root=args.data_dir,
                train=False,
                transform=transform_test,
                download=True)

    elif args.class_num == 100:
        train_set = torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=True,
            transform=transform_train,
            download=True)

        test_set = torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=False,
            transform=transform_test,
            download=True)
    else:
        raise NotImplementedError(args.class_num)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    out_dir = f'{args.neu}_e{args.epochs}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    if args.amp:
        out_dir += '_amp'
    if args.readout != 'trunk_distal':
        out_dir += f'_ro-{args.readout}'
        if args.readout == 'linear_act':
            out_dir += f'-{args.readout_act}'
    if (args.L, args.U) != (-4, 4):
        out_dir += f'_L{args.L}U{args.U}'
    if args.soma_kernel != 'free':
        out_dir += f'_soma-{args.soma_kernel}'
    if args.proximal != 'fastest':
        out_dir += f'_prox-{args.proximal}'
    if args.coupling != 'none':
        out_dir += f'_coup-{args.coupling}'
    if args.dmpr_ablation != 'none':
        out_dir += f'_abl-{args.dmpr_ablation}'



    if args.P is not None:
        out_dir += f'_P{args.P}'
        if args.exp_init:
            out_dir += '_ei'

    pt_dir = os.path.join(args.out_dir, 'pt', out_dir)
    out_dir = os.path.join(args.out_dir, out_dir)

    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)

    #net = CIFAR10Net(channels=args.channels, neu=args.neu, T=32, class_num=args.class_num, P=args.P, exp_init=args.exp_init)
    net = CIFAR10Net(channels=args.channels, class_num=args.class_num, readout=args.readout, readout_act=args.readout_act, L=args.L, thre=args.U, soma_kernel=args.soma_kernel, coupling=args.coupling, proximal=args.proximal, dmpr_ablation=args.dmpr_ablation)  ##
    net.to(args.device)
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    if args.dmpr_stats or args.distal_stim or args.spike_stats:
        # Rebuttal analyses of a trained model; no training is run
        import distal_stimulation
        ckpt_paths = {os.path.abspath(p) for p in (args.dmpr_stats, args.distal_stim, args.spike_stats) if p}
        if len(ckpt_paths) > 1:
            raise ValueError('-dmpr-stats, -distal-stim and -spike-stats must point to the same checkpoint')
        ckpt_path = ckpt_paths.pop()
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        net.load_state_dict(checkpoint['net'])

        def make_batches():
            for img, _ in test_data_loader:
                yield img.to(args.device, non_blocking=True)

        if args.dmpr_stats:  # Table R3-2, whole test set
            stats = distal_stimulation.run_dmpr_statistics(net, make_batches)
            distal_stimulation.save_and_print_stats(stats, os.path.join(os.path.dirname(ckpt_path), 'dmpr_stats.json'))
        if args.distal_stim:  # Table R2-1
            result = distal_stimulation.run_distal_stimulation(
                net, make_batches, max_samples=args.stim_samples,
                fraction=args.stim_fraction, n_subsets=args.stim_subsets,
            )
            distal_stimulation.save_and_print(result, os.path.join(os.path.dirname(ckpt_path), 'distal_stim.json'))
        if args.spike_stats:  # Table R4-5
            import spike_stats
            result = spike_stats.run_spike_statistics(net, make_batches, max_samples=args.spike_samples)
            spike_stats.save_and_print(result, os.path.join(os.path.dirname(ckpt_path), 'spike_stats.json'))
        if os.path.isdir(pt_dir) and not os.listdir(pt_dir):
            os.rmdir(pt_dir)  # created above for training only
        return

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise NotImplementedError(args.opt)
    
    warmup_epochs = max(0, min(args.warmup_epochs, args.epochs - 1))
    if warmup_epochs > 0:
        warmup_start_factor = max(args.warmup_lr / args.lr, 1e-8)
        warmup_start_factor = min(warmup_start_factor, 1.0)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - warmup_epochs,
            eta_min=args.min_lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )

    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        print(max_test_acc)



    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))



    for epoch in range(start_epoch, args.epochs):
        if args.neu == 'if5pmd8':
            for m in net.modules():
                if isinstance(m, (DecayMaskedLinear, DecayPorderMaskedLinear)):
                    mk = epoch / (args.epochs - 1)
                    m.k = min(mk * 8, 1.)


        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        for batch_index, (img, label) in enumerate(train_data_loader):
            optimizer.zero_grad()
            img = img.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)



            with torch.cuda.amp.autocast(enabled=scaler is not None):
                y = net(img)
                loss = F.cross_entropy(y, label, label_smoothing=0.1)


            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_samples += label.shape[0]
            train_loss += loss.item() * label.shape[0]
            train_acc += (y.argmax(1) == label.argmax(1)).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                y = net(img)
                loss = F.cross_entropy(y, label)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (y.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        #print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main() 