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
from torch.cuda import amp
from utils import *
import os
import time
import json
import argparse
import logging
import h5py
from datetime import datetime
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
from functools import partial
import sys
from ssc_dataset import (
    create_asrc_style_ssc_dataloader,
    create_spikingjelly_frame_dataloader,
    create_fixed_time_h5_dataloader,
)
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
from module import dendrite,dend_compartment,soma,neuron,wiring
from spikingjelly.activation_based import functional, surrogate, layer
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from module.dend_compartment import ChannelPreservingTrunkDistalDendCompartment,SparseChannelPreservingTrunkDistalDendCompartment,CoupledSparseChannelPreservingTrunkDistalDendCompartment
from  module.soma import DecayPorderMaskedLinear

def setup_train_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def scalar_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return float(value)

#device = 'cuda'
class ff_SSC(nn.Module):
    def __init__(self, in_dim=700, hidden=[128, 128, 128], out_dim=35,drop=0.1,T=250):  ##
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, hidden[0], bias=True),
                   #layer.BatchNorm1d(hidden[0]),
                   nn.Dropout(drop),
                   SparseChannelPreservingTrunkDistalDendCompartment(channels=hidden[0],num_branches=8,compartments_per_branch=4,branch_degree=1,learn_comp_gain=True,learn_edge_gain=True,merge_norm='mean'), #,branch_readout_mode="linear"),#,learn_edge_gain=False,learn_comp_gain=False),
                   #nn.Identity(),
                   #soma.AstroMaskedSlidingPSN(order=T,exp_init=True,astro_update_interval=1,astro_gain=0.2)]
                   soma.MaskedSlidingPSN(order=T,surrogate_function=surrogate.Sigmoid(),exp_init=True)]
                   #soma.LIFNODE()]
                   #soma.IFNode5PorderMaskD(T=T,P=250)]
                   #soma.IntergerSoma_ssf(decay_input=False)]
                   #soma.PSNIntergerSoma_ssf(psn_order=T,psn_exp_init=True,thre=4)]
                   #soma.SelectiveAstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=25,astro_pool_kernel=5)]
                   #soma.IFNode5(T=T)]
                   #soma.AstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=5,astro_gain=0.2,astro_lambda=0.05,astro_trace_decay=0.8,astro_pool_kernel=5,astro_thre=0.8)] 
                   #soma.AstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=5,astro_pool_kernel=9,astro_channel_pool_when_length1=True,astro_thre=0.2)]  #需要试astro_event_write=True
        layers += [nn.Linear(hidden[0], hidden[1], bias=True),
                   #layer.BatchNorm1d(hidden[1]),
                   nn.Dropout(drop),
                   SparseChannelPreservingTrunkDistalDendCompartment(channels=hidden[1],num_branches=8,compartments_per_branch=4,branch_degree=1,learn_comp_gain=True,learn_edge_gain=True,merge_norm='mean'),  #,branch_readout_mode="linear"), #,learn_edge_gain=False,learn_comp_gain=False),
                   #nn.Identity(),
                   #soma.AstroMaskedSlidingPSN(order=T,exp_init=True,astro_update_interval=1,astro_gain=0.2)]
                   soma.MaskedSlidingPSN(order=T,surrogate_function=surrogate.Sigmoid(),exp_init=True)]
                   #soma.LIFNODE()]
                   #soma.IFNode5PorderMaskD(T=T,P=250)]
                   #soma.IntergerSoma_ssf(decay_input=False)]
                   #soma.PSNIntergerSoma_ssf(psn_order=T,psn_exp_init=True,thre=4)]
                   #soma.SelectiveAstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=25,astro_pool_kernel=5)]
                   #soma.IFNode5(T=T)]
                   #soma.AstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=5,astro_gain=0.2,astro_lambda=0.05,astro_trace_decay=0.8,astro_pool_kernel=5,astro_thre=0.8)] 
                   #soma.AstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=5,astro_pool_kernel=9,astro_channel_pool_when_length1=True,astro_thre=0.2)]
        layers += [nn.Linear(hidden[1], hidden[2], bias=True),
                   #layer.BatchNorm1d(hidden[1]),
                   nn.Dropout(drop),
                   SparseChannelPreservingTrunkDistalDendCompartment(channels=hidden[2],num_branches=8,compartments_per_branch=4,branch_degree=1,learn_comp_gain=True,learn_edge_gain=True,merge_norm='mean'),  #,branch_readout_mode="linear"), #,learn_edge_gain=False,learn_comp_gain=False),
                   #nn.Identity(),
                   #soma.AstroMaskedSlidingPSN(order=T,exp_init=True,astro_update_interval=1,astro_gain=0.2)]
                   soma.MaskedSlidingPSN(order=T,surrogate_function=surrogate.Sigmoid(),exp_init=True)]
                   #soma.LIFNODE()]
                   #soma.IFNode5PorderMaskD(T=T,P=250)]
                   #soma.IntergerSoma_ssf(decay_input=False)]
                   #soma.PSNIntergerSoma_ssf(psn_order=T,psn_exp_init=True,thre=4)]
                   #soma.SelectiveAstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=25,astro_pool_kernel=5)]
                   #soma.IFNode5(T=T)]
                   #soma.AstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=5,astro_gain=0.2,astro_lambda=0.05,astro_trace_decay=0.8,astro_pool_kernel=5,astro_thre=0.8)] 
                   #soma.AstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=5,astro_pool_kernel=9,astro_channel_pool_when_length1=True,astro_thre=0.2)]

        layers += [nn.Linear(hidden[2], out_dim,bias=True)]
                   #nn.Dropout(drop),  ##
                   #soma.IFNode5(T=T)] ##
        self.features = nn.Sequential(*layers)

        functional.set_step_mode(self, 'm')

    def forward(self, x): # x [N, T, F]
        x = x.transpose(0, 1)
        #x = x.unsqueeze(-1)
        assert x.dim() == 3, "dimension of x is not correct!"

        #x = self.features(x).squeeze(-1)
        x = self.features(x)
        return x.mean(0) ###


def _expand_soma_skip(skip, num_layers, soma_type):
    if skip is None:
        default_skip = 3 if soma_type == "src" else 6
        return [default_skip] * num_layers

    skip = [int(value) for value in skip]
    if len(skip) == 1:
        skip = skip * num_layers
    if len(skip) != num_layers:
        raise ValueError(
            f"soma_skip must have length 1 or {num_layers}, got {len(skip)}"
        )
    if any(value <= 0 for value in skip):
        raise ValueError("all soma_skip values must be positive")
    return skip


class ff_SSC_SRC_ASRC(nn.Module):
    """ASRC-SNN's feedforward model with the existing Dend kept in each block."""

    def __init__(
        self,
        in_dim=700,
        hidden=(128, 128, 128), ##
        out_dim=35,
        drop=0.1,
        model_version="SRC",
        skip=None,
        decay_para=0.5,
        threshold=1.0,
        hard_reset=False,
        detach_reset=False,
        de_tau=0.96,
    ):
        super().__init__()
        hidden = [int(channels) for channels in hidden]
        if not hidden or any(channels <= 0 for channels in hidden):
            raise ValueError("hidden must contain positive channel counts")
        model_version = model_version.upper()
        if model_version not in {"SRC", "ASRC"}:
            raise ValueError("model_version must be 'SRC' or 'ASRC'")

        skip_per_layer = _expand_soma_skip(
            skip, len(hidden), soma_type=model_version.lower()
        )
        spiking_neuron = partial(
            soma.LIFNODE,
            decay_para=decay_para,
            v_threshold=threshold,
            surrogate_function=soma.Triangle.apply,
            hard_reset=hard_reset,
            detach_reset=detach_reset,
        )

        layers = []
        current_channels = int(in_dim)
        for layer_index, next_channels in enumerate(hidden):
            layers.extend([
                nn.Linear(current_channels, next_channels, bias=True),
                layer.Dropout(drop, step_mode="m"),
                SparseChannelPreservingTrunkDistalDendCompartment(
                    channels=next_channels,
                    num_branches=8,
                    compartments_per_branch=6,
                    branch_degree=1,
                    learn_comp_gain=True,
                    learn_edge_gain=True,
                    merge_norm="mean",
                ),
            ])

            if model_version == "SRC":
                soma_layer = soma.skip_RecurrentContainer(
                    sub_module=spiking_neuron(),
                    hid_dim=next_channels,
                    skip=skip_per_layer[layer_index],
                    step_mode="m",
                )
            else:
                #soma_layer = soma.adskip_RecurrentContainer(
                #    sub_module=spiking_neuron(),
                #    hid_dim=next_channels,
                #    skip=skip_per_layer[layer_index],
                #    de_tau=de_tau,
                #    step_mode="m",
                #)

                #soma_layer = soma.LIFSoma(v_reset=None,step_mode='m')
                #soma_layer = soma.PSNIntergerSoma_ssf(psn_order=250,psn_exp_init=True)
                soma_layer = soma.MaskedSlidingPSN(order=250,exp_init=True,surrogate_function=soma.Triangle.apply)
            layers.append(soma_layer)
            current_channels = next_channels

        layers.append(nn.Linear(current_channels, out_dim, bias=True))
        self.features = nn.Sequential(*layers)
        self.model_version = model_version
        self.skip = tuple(skip_per_layer)

    def forward(self, x):
        # Input [N, T, F] -> multi-step network input [T, N, F].
        x = x.transpose(0, 1)
        if x.dim() != 3:
            raise ValueError("Expected SSC input shape [N, T, F]")
        x = self.features(x)
        return x.sum(0)  ##

    def decrease_tau(self):
        for module in self.features:
            if isinstance(module, soma.adskip_RecurrentContainer):
                module.decrease_tau()


class ff_SHD(nn.Module):
    def __init__(self, in_dim=700, hidden=256, out_dim=20, drop=0.2,T=250):  ##
        super().__init__()
        layers = []
        layers += [layer.Conv1d(in_dim, hidden,kernel_size=1),
                   #layer.BatchNorm1d(hidden[0]),
                   nn.Dropout(drop),
                   SparseChannelPreservingTrunkDistalDendCompartment(channels=hidden,num_branches=10,compartments_per_branch=6,branch_degree=1,learn_comp_gain=True,learn_edge_gain=True,merge_norm='mean'), #,branch_readout_mode="linear"),#,learn_edge_gain=False,learn_comp_gain=False),
                   #nn.Identity(),
                   #soma.AstroMaskedSlidingPSN(order=T,exp_init=True,astro_update_interval=25)]
                   soma.MaskedSlidingPSN(order=T,surrogate_function=surrogate.Sigmoid(),exp_init=True)]
                   #soma.IFNode5PorderMaskD(T=T,P=250)]
                   #soma.IntergerSoma_ssf(decay_input=False)]
                   #soma.PSNIntergerSoma_ssf(psn_order=T)]
                   #soma.SelectiveAstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=25,astro_pool_kernel=5)]
                   #soma.IFNode5(T=T)]
                   #soma.AstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=5,astro_gain=0.2,astro_lambda=0.05,astro_trace_decay=0.8,astro_pool_kernel=5,astro_thre=0.8)] 
                   #soma.AstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=5,astro_pool_kernel=9,astro_channel_pool_when_length1=True,astro_thre=0.2)]  #需要试astro_event_write=True
        layers += [layer.Conv1d(hidden, out_dim,kernel_size=1)]
                   #nn.Dropout(drop),  ##
                   #soma.IFNode5(T=T)] ##
        self.features = nn.Sequential(*layers)

        functional.set_step_mode(self, 'm')

    def forward(self, x): # x [N, T, F]
        x = x.transpose(0, 1)
        x = x.unsqueeze(-1)
        assert x.dim() == 4, "dimension of x is not correct!"

        x = self.features(x).squeeze(-1)
        return x.mean(0)    

class DendSomaResidualBlock(nn.Module):
    def __init__(self, channels, drop=0.0, T=250):
        super().__init__()
        self.dend1 = SparseChannelPreservingTrunkDistalDendCompartment(
            channels=channels,
            num_branches=8,
            compartments_per_branch=4,
            branch_degree=1,
            merge_norm='mean'
        )
        self.dend2 = SparseChannelPreservingTrunkDistalDendCompartment(
            channels=channels,
            num_branches=8,
            compartments_per_branch=4,
            branch_degree=1,
            merge_norm='mean'
        )
        self.conv1 = layer.Conv1d(channels, channels, kernel_size=1)
        #self.bn1 = layer.BatchNorm1d(channels)
        self.drop = nn.Dropout(drop)

        self.soma1 = soma.PSNIntergerSoma_ssf(
            psn_order=T,
            psn_exp_init=True
        )
        self.soma2 = soma.PSNIntergerSoma_ssf(
            psn_order=T,
            psn_exp_init=True
        )
        self.conv2 = layer.Conv1d(channels, channels, kernel_size=1)
        #self.bn2 = layer.BatchNorm1d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.dend1(out)
        out = self.soma1(out)
        out = out + identity
        identity = out

        out = self.conv2(out)
        out = self.dend2(out)
        out = self.soma2(out)
        #out = self.bn1(out)
        #out = self.drop(out)
        out = out + identity
        #out = self.bn2(out)

        return out

class ff_SHD_res(nn.Module):
    def __init__(self, in_dim=700, hidden=128, out_dim=20, drop=0.0, T=250, depth=1):
        super().__init__()

        self.stem = layer.Conv1d(in_dim, hidden, kernel_size=1)

        self.blocks = nn.Sequential(*[
            DendSomaResidualBlock(hidden, drop=drop, T=T)
            for _ in range(depth)
        ])

        self.head = layer.Conv1d(hidden, out_dim, kernel_size=1)

        functional.set_step_mode(self, 'm')

    def forward(self, x):  # [N, T, F]
        x = x.transpose(0, 1)  # [T, N, F]
        x = x.unsqueeze(-1)    # [T, N, F, 1]

        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x).squeeze(-1)

        return x.mean(0)

def train(train_loader, model, criterion, optimizer, epoch, args, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(gpu, dtype=torch.float32, non_blocking=True)
        target = target.to(gpu, non_blocking=True)

        functional.reset_net(model)
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.print_freq > 0 and i % args.print_freq == 0:
            progress.display(i)
    logging.info(
        'Train Epoch: [{}/{}], lr: {:.6f}, top1: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'],
                                                                scalar_value(top1.avg)))
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(gpu, dtype=torch.float32, non_blocking=True)
            target = target.to(gpu, non_blocking=True)

            functional.reset_net(model)
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # torch.cuda.synchronize()

            if args.print_freq > 0 and i % args.print_freq == 0:
                progress.display(i)

        logging.info(
            ' * Loss {:.4e} Acc@1 {:.3f} Acc@5 {:.3f}'.format(
                losses.avg,
                scalar_value(top1.avg),
                scalar_value(top5.avg),
            )
        )

    return top1.avg, top5.avg, losses.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
class SpikeIterator:
    def __init__(self, X, y, batch_size, nb_steps, nb_units, max_time, n_bins = 1,shuffle=True,device='cuda:0'):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        # self.max_time = max_time
        self.shuffle = shuffle
        self.labels_ = np.array(y, dtype=np.float32)
        self.num_samples = len(self.labels_)
        self.number_of_batches = np.ceil(self.num_samples / self.batch_size)
        self.sample_index = np.arange(len(self.labels_))
        # compute discrete firing times
        self.firing_times = X['times']
        self.units_fired = X['units']
        self.time_bins = np.linspace(0, max_time, num=nb_steps)

        self.n_bins = n_bins

        self.device = device
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.number_of_batches)

    def __next__(self):
        if self.counter < self.number_of_batches:
            batch_index = self.sample_index[
                          self.batch_size * self.counter:min(self.batch_size * (self.counter + 1), self.num_samples)]
            coo = [[] for i in range(3)]
            for bc, idx in enumerate(batch_index):
                times = np.digitize(self.firing_times[idx], self.time_bins)
                units = self.units_fired[idx]
                batch = [bc for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.LongTensor(coo).to(self.device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(self.device)

            X_batch = torch.sparse.FloatTensor(i, v, torch.Size(
                [len(batch_index), self.nb_steps, self.nb_units])).to_dense().to(
                self.device)
            
            ###############################################################
            binned_len = X_batch.shape[-1]//self.n_bins
            binned_frames = torch.zeros((len(batch_index), self.nb_steps, binned_len)).to(
                self.device)
            for i in range(binned_len):
                binned_frames[:,:,i] = X_batch[:, :,self.n_bins*i : self.n_bins*(i+1)].sum(axis=-1)
            ###############################################################
            y_batch = torch.tensor(self.labels_[batch_index], device=self.device).long()

            X_batch = binned_frames
            self.counter += 1
            return X_batch.to(device=self.device), y_batch.to(device=self.device)

        else:
            raise StopIteration

def getData(root, dataset):
    dataset = dataset
    root_path = root  + '/extract'
    train_file = h5py.File(os.path.join(root_path, dataset.lower()+'_train.h5'), 'r')
    test_file = h5py.File(os.path.join(root_path, dataset.lower()+'_test.h5'), 'r')

    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']
    return (x_train, y_train), (x_test, y_test)


# python train_ssc.py --task SSC --device cuda:6 --lr 0.005 --wd 5e-4 --cos --ssc-preprocess asrc --ssc-n-bins 1       --epochs 400 --workers 16 --cos   --optim sgd    --lr 0.1
# python train_ssc.py --task SSC --device cuda:6 --lr 0.005 --wd 5e-4 --cos --ssc-preprocess asrc --ssc-n-bins 1
# python train_ssc.py --task SHD --device cuda:1 --lr 0.005 --data-root /data/hyx/ViT-dend/data/shd/data_shd --wd 5e-4 --cos

parser = argparse.ArgumentParser(description='Sequential SHD/SSC')
parser.add_argument('--task', default='SSC', type=str, help='SHD, SSC')
parser.add_argument('--optim', default=None, choices=['sgd', 'adam', 'adamw'],
                    help='optimizer; SRC/ASRC SSC defaults to the paper setting Adam')
parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
parser.add_argument('-p', '--print-freq', default=512, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')
parser.add_argument('--epochs', default=None, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=None, type=float, metavar='LR', help='maximum learning rate',
                    dest='lr')
parser.add_argument('--schedule', default=[], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--batch-size', default=None, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=None, type=float, metavar='W', help='weight decay')
parser.add_argument('--drop', default=None, type=float, help='hidden-layer dropout probability')
parser.add_argument("--workers", type=int, default=8)
parser.add_argument('--cos', action='store_true', default=False, help='use cosine lr schedule')
parser.add_argument('--scheduler', default=None, choices=['onecycle', 'cos', 'step'],
                    help='learning-rate scheduler; SRC/ASRC SSC defaults to OneCycle')
parser.add_argument('--data-root', default='/data2/hyx/ViT-dend/data/ssc', type=str,
                    help='path to extract/ or frames_number_250_split_by_number/')
parser.add_argument('--ssc-preprocess', default=None, choices=['current', 'asrc'],
                    help='SSC preprocessing pipeline: current fixed-time bins or ASRC-SNN style')
parser.add_argument('--ssc-n-bins', default=5, type=int,
                    help='channel bin size for ASRC-SNN style SSC preprocessing')
parser.add_argument('--ssc-soma', '--model-version', dest='ssc_soma',
                    default='psn', type=str.lower, choices=['psn', 'src', 'asrc'],
                    help='SSC soma: current parallel PSN, fixed-skip SRC-LIF, or adaptive-skip ASRC-LIF')
parser.add_argument('--soma-skip', '--skip', dest='soma_skip', default=None,
                    nargs='+', type=int,
                    help='SRC delay or ASRC maximum delay; one value broadcasts to all hidden layers')
parser.add_argument('--soma-decay-para', '--decay-para', dest='soma_decay_para',
                    default=0.5, type=float,
                    help='LIF membrane decay alpha in v[t] = alpha * v[t-1] + input[t]')
parser.add_argument('--soma-v-threshold', '--threshold', dest='soma_v_threshold',
                    default=1.0, type=float,
                    help='SRC/ASRC LIF firing threshold')
parser.add_argument('--soma-hard-reset', '--hard-reset', dest='soma_hard_reset',
                    action='store_true', default=False,
                    help='use hard reset in the SRC/ASRC LIF soma (default: soft reset)')
parser.add_argument('--soma-detach-reset', '--detach-reset', dest='soma_detach_reset',
                    action='store_true', default=False,
                    help='detach the SRC/ASRC LIF reset from the backward graph')
parser.add_argument('--asrc-de-tau', '--de-tau', dest='asrc_de_tau',
                    default=0.96, type=float,
                    help='multiplicative ASRC temperature decay applied after each epoch')
parser.add_argument('--asrc-lr-skip', '--lr-skip', dest='asrc_lr_skip',
                    default=0.1, type=float,
                    help='learning rate for ASRC skip logits (no weight decay)')
parser.add_argument('--device', default=None, type=str, help="device, e.g. 'cuda', 'cuda:5', or 'cpu'")
parser.add_argument('--no-pin-memory', action='store_true', default=False, help='disable DataLoader pinned memory')

args = parser.parse_args()
paper_aligned_soma = (
    args.task.upper() == 'SSC' and args.ssc_soma in {'src', 'asrc'}
)
args.training_profile = 'asrc-paper-except-dend' if paper_aligned_soma else 'legacy'

if args.epochs is None:
    args.epochs = 100 if paper_aligned_soma else 200
if args.lr is None:
    args.lr = 1e-3 if paper_aligned_soma else 1e-2
if args.batch_size is None:
    args.batch_size = 128 if paper_aligned_soma else 32
if args.wd is None:
    args.wd = 0.0
if args.drop is None:
    args.drop = 0.1 
if args.optim is None:
    # The legacy code labelled this choice "adam" but instantiated AdamW.
    args.optim = 'adam' if paper_aligned_soma else 'adamw'
if args.ssc_preprocess is None:
    args.ssc_preprocess = 'asrc' if paper_aligned_soma else 'current'
if args.cos:
    if args.scheduler not in {None, 'cos'}:
        raise ValueError("--cos conflicts with a non-cosine --scheduler")
    args.scheduler = 'cos'
elif args.scheduler is None:
    args.scheduler = 'onecycle' if paper_aligned_soma else 'step'
args.cos = args.scheduler == 'cos'

if args.lr <= 0.0:
    raise ValueError("--lr must be positive")
if args.epochs <= 0:
    raise ValueError("--epochs must be positive")
if args.batch_size <= 0:
    raise ValueError("--batch-size must be positive")
if not 0.0 <= args.drop < 1.0:
    raise ValueError("--drop must be in [0, 1)")
if args.wd < 0.0:
    raise ValueError("--wd must be non-negative")
if not 0.0 <= args.soma_decay_para < 1.0:
    raise ValueError("--decay-para must be in [0, 1)")
if args.soma_v_threshold <= 0.0:
    raise ValueError("--threshold must be positive")
if not 0.0 < args.asrc_de_tau <= 1.0:
    raise ValueError("--de-tau must be in (0, 1]")
if args.asrc_lr_skip <= 0.0:
    raise ValueError("--asrc-lr-skip must be positive")
if args.results_dir == '':
    args.results_dir = '/data3/hyx/NR-DN/exp/'+args.task+'ff'+'-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

Path(args.results_dir).mkdir(parents=True, exist_ok=True)
logger = setup_train_logging(os.path.join(args.results_dir, "log-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"))

if args.device is not None:
    gpu = torch.device(args.device)
elif torch.cuda.is_available():
    gpu = torch.device('cuda')
    print('GPU is available')
else:
    gpu = torch.device('cpu')
    print('GPU is not available')
seed_everything(seed=args.seed, is_cuda=True)

torch.backends.cudnn.benchmark = not paper_aligned_soma

if args.task.upper() == 'SSC' or args.task.upper() == 'SHD':
    T = 250
    max_time = 1.4
    out_dim = 35 if args.task.upper() == 'SSC' else 20
    pin_memory = (not args.no_pin_memory) and gpu.type == 'cuda'
    if args.ssc_preprocess == 'asrc':
        if args.task.upper() != 'SSC':
            raise ValueError("--ssc-preprocess asrc is only intended for SSC")
        if 700 % args.ssc_n_bins != 0:
            raise ValueError("700 must be divisible by --ssc-n-bins")
        in_dim = 700 // args.ssc_n_bins
        (x_train, y_train), (x_test, y_test) = getData(args.data_root, args.task)
        train_loader = SpikeIterator(
                x_train, y_train, args.batch_size, T, 700, max_time,
                n_bins=args.ssc_n_bins, shuffle=True, device=args.device
            )
        test_loader = SpikeIterator(
                x_test, y_test, args.batch_size, T, 700, max_time,
                n_bins=args.ssc_n_bins, shuffle=False, device=args.device
            )
    else:
        in_dim = 700
        train_loader = create_fixed_time_h5_dataloader(
            split="train",
            batch_size=args.batch_size,
            root_path=args.data_root,
            nb_steps=250,
            nb_units=700,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin_memory,
        )

        test_loader = create_fixed_time_h5_dataloader(
            split="test",
            batch_size=args.batch_size,
            root_path=args.data_root,
            nb_steps=250,
            nb_units=700,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin_memory,
        )
else:
    raise NotImplementedError

if args.task.upper() == 'SHD':
    #model = ff_SHD(in_dim=in_dim, out_dim=out_dim,T=T).to(gpu)
    model = ff_SHD_res(in_dim=in_dim,out_dim=out_dim,T=T).to(gpu)    
else:
    if args.ssc_soma == 'psn':
        model = ff_SSC(
            in_dim=in_dim, out_dim=out_dim, drop=args.drop, T=T
        ).to(gpu)
    else:
        model = ff_SSC_SRC_ASRC(
            in_dim=in_dim,
            out_dim=out_dim,
            drop=args.drop,
            model_version=args.ssc_soma,
            skip=args.soma_skip,
            decay_para=args.soma_decay_para,
            threshold=args.soma_v_threshold,
            hard_reset=args.soma_hard_reset,
            detach_reset=args.soma_detach_reset,
            de_tau=args.asrc_de_tau,
        ).to(gpu)
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"number of params: {n_parameters}")
asrc_soma_modules = [
    module for module in model.modules()
    if isinstance(module, soma.adskip_RecurrentContainer)
]

logging.info(str(model))

criterion = nn.CrossEntropyLoss().to(gpu)

skip_parameters = [module.skip_para for module in asrc_soma_modules]
special_parameter_ids = {id(parameter) for parameter in skip_parameters}
base_parameters = [
    parameter for parameter in model.parameters()
    if parameter.requires_grad and id(parameter) not in special_parameter_ids
]
optimizer_groups = [{
    'params': base_parameters,
    'lr': args.lr,
    'weight_decay': args.wd,
    'lr_scale': 1.0,
    'group_name': 'base',
}]
if skip_parameters:
    optimizer_groups.append({
        'params': skip_parameters,
        'lr': args.asrc_lr_skip,
        'weight_decay': 0.0,
        'lr_scale': args.asrc_lr_skip / args.lr,
        'group_name': 'skip',
    })

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(optimizer_groups, lr=args.lr, momentum=0.9)
elif args.optim == 'adam':
    optimizer = torch.optim.Adam(optimizer_groups, lr=args.lr)
elif args.optim == 'adamw':
    optimizer = torch.optim.AdamW(optimizer_groups, lr=args.lr)
else:
    raise NotImplementedError

if args.scheduler == 'onecycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group['lr'] for group in optimizer.param_groups],
        total_steps=args.epochs,
        pct_start=0.3  ###
    )
else:
    scheduler = None

# dump args
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)
logging.info(str(args))
start_epoch = 0
writer = SummaryWriter(args.results_dir, purge_step=start_epoch)

if args.print_freq > len(train_loader):
    args.print_freq = max(1, math.ceil(len(train_loader) / 2))

best_acc = argparse.Namespace(top1=0, top5=0)
for epoch in range(start_epoch, args.epochs):
    for m in model.modules():
        if isinstance(m, DecayPorderMaskedLinear):
            mk = epoch / (args.epochs - 1)
            m.k = min(mk * 8, 1.)
    epoch_start_time = time.time()
    if scheduler is None:
        adjust_learning_rate(optimizer, epoch, args)
        for parameter_group in optimizer.param_groups:
            parameter_group['lr'] *= parameter_group.get('lr_scale', 1.0)
    if args.task == 'SSC':
        train_loader.reset()
        test_loader.reset()
    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, args,gpu=gpu)
    #train_loader.reset()
    acc1, acc5, test_loss = validate(test_loader, model, criterion, args,gpu=gpu)
    #test_loader.reset()
    if scheduler is not None:
        # ASRC-SNN advances OneCycle once after each complete train/eval epoch.
        scheduler.step()

    train_acc_value = scalar_value(train_acc)
    test_acc_value = scalar_value(acc1)
    test_acc5_value = scalar_value(acc5)
    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_acc', train_acc_value, epoch)
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_acc', test_acc_value, epoch)
    writer.add_scalar('test_acc5', test_acc5_value, epoch)
    writer.add_scalar('lr', lr, epoch)
    for parameter_group in optimizer.param_groups:
        writer.add_scalar(
            f"lr/{parameter_group.get('group_name', 'unnamed')}",
            parameter_group['lr'],
            epoch,
        )
    for layer_index, asrc_soma in enumerate(asrc_soma_modules):
        writer.add_scalar(
            f'asrc/layer_{layer_index}_tau',
            scalar_value(asrc_soma.tau),
            epoch,
        )
        soft_weights = F.softmax(
            asrc_soma.skip_para / asrc_soma.tau, dim=0
        ).detach().flatten().cpu()
        writer.add_scalar(
            f'asrc/layer_{layer_index}_selected_skip',
            int(torch.argmax(soft_weights).item()) + 1,
            epoch,
        )
        for lag_index, lag_weight in enumerate(soft_weights, start=1):
            writer.add_scalar(
                f'asrc/layer_{layer_index}_lag_{lag_index}',
                float(lag_weight),
                epoch,
            )

    is_best = test_acc_value > best_acc.top1
    best_acc.top1 = max(best_acc.top1, test_acc_value)
    best_acc.top5 = max(best_acc.top5, test_acc5_value)

    logging.info(
        'epoch = {}, train_loss = {:.4f}, train_acc = {:.4f}, '
        'test_loss = {:.4f}, test_acc = {:.4f}, best_test_acc = {:.4f}, '
        'lr = {:.6f}, time = {:.2f}s'.format(
            epoch,
            train_loss,
            train_acc_value,
            test_loss,
            test_acc_value,
            best_acc.top1,
            lr,
            time.time() - epoch_start_time,
        )
    )
    if asrc_soma_modules:
        logging.info(
            'ASRC state: tau = {}, selected_skip = {}'.format(
                [round(scalar_value(module.tau), 6) for module in asrc_soma_modules],
                [
                    int(torch.argmax(module.skip_para.detach()).item()) + 1
                    for module in asrc_soma_modules
                ],
            )
        )

    if asrc_soma_modules:
        model.decrease_tau()
    #print('current beta values:', model)

    save_checkpoint({
        'epoch': epoch + 1,
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
    }, is_best=is_best, dirname=args.results_dir, filename='checkpoint.pth.tar')

writer.close()