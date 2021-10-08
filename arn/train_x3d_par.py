import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import logging
import sys
import torchvision
from torchvision import datasets, transforms
from torchsummary import summary

import numpy as np
import tqdm
from utils.apmeter import APMeter

import x3d as resnet_x3d

from data.par_data import customized_dataset, PAR

from transforms.spatial_transforms_old import Compose, Normalize, \
    RandomHorizontalFlip, MultiScaleRandomCrop, \
    MultiScaleRandomCropMultigrid, ToTensor, \
    CenterCrop, CenterCropScaled
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel
import pdb

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)
parser.add_argument('-task', default='class', type=str)
parser.add_argument('-config', default="config.txt", type=str)
parser.add_argument('-id', default="", type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
ID = args.id
BS = 4
BS_UPSCALE = 2
INIT_LR = 0.0002 * BS_UPSCALE
GPUS = 1

X3D_VERSION = 'M'

with open(args.config, "r") as f:
    raw_lines = f.readlines()
lines = []
for x in raw_lines:
    if x[0] != "#":
        lines.append(x)
TA2_ROOT = lines[0][:-1]
TA2_ANNO = lines[1][:-1]
TA2_VAL = lines[2][:-1]
model_save_path = lines[3][:-1]
save_txt_dir = lines[4][:-1]


# TODO: Why are these here? We can take the len() of the dataset
# TA2_DATASET_SIZE = {'train':13446, 'val':1491}
# TA2_DATASET_SIZE = {'train':4235, 'val':471} # UCF101 TA2 split
# TA2_DATASET_SIZE = {'train':1906, 'val':212} # HMDB51 TA2 split
TA2_MEAN = [0, 0, 0]
TA2_STD = [1, 1, 1]

# warmup_steps=0
def run(init_lr=INIT_LR, max_epochs=100, root=TA2_ROOT, anno=TA2_ANNO, val_anno=TA2_VAL, batch_size=BS*BS_UPSCALE, task="class"):

    frames=80 # DOUBLED INSIDE DATASET, AS LONGER CLIPS
    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,256.], 'XL':[360.,450.]}[X3D_VERSION] # 'M':[256.,320.] FOR LONGER SCHEDULE
    gamma_tau = {'S':6, 'M':5, 'XL':5}[X3D_VERSION] # DOUBLED INSIDE DATASET, AS LONGER CLIPS

    st_steps = 0 # FOR LR WARM-UP
    load_steps = 0 # FOR LOADING AND PRINT SCHEDULE
    steps = 0
    epochs = 0



    train_spatial_transforms = Compose([MultiScaleRandomCropMultigrid([crop_size/i for i in resize_size], crop_size),
                                        RandomHorizontalFlip(),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD)])

    val_spatial_transforms = Compose([CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD)])

    dataset = PAR(split_file=anno,
                     task="class",
                     split='training',
                     root=root,
                     spatial_transform=train_spatial_transforms,
                     frames=80,
                     gamma_tau=gamma_tau,
                     crops=1,
                     prediction_target = task)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=16,
                                             pin_memory=True)

    val_dataset = PAR(split_file=val_anno,
                         split='validation',
                         root=root,
                         task="class",
                         spatial_transform=val_spatial_transforms,
                         frames=80,
                         gamma_tau=gamma_tau,
                         crops=10,
                         prediction_target=task)



    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size//2,
                                                 shuffle=False,
                                                 num_workers=16,
                                                 pin_memory=True)
    num_steps_per_update = 1 # ACCUMULATE GRADIENT IF NEEDED
    cur_iterations = steps * num_steps_per_update
    iterations_per_epoch = len(dataset)//batch_size
    val_iterations_per_epoch = len(val_dataset)//(batch_size//2)
    max_steps = iterations_per_epoch * max_epochs


    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('train',len(datasets['train']),'val',len(datasets['val']))
    print('Total iterations:', max_steps, 'Total epochs:', max_epochs)
    print('datasets created')


    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=400, n_input_channels=3, dropout=0.5, base_bn_splits=1)
    load_ckpt = torch.load('models/x3d_multigrid_kinetics_fb_pretrained.pt')
    x3d.load_state_dict(load_ckpt['model_state_dict'])
    save_model = model_save_path + '/x3d_PAR_rgb_ADAM_'+task+"_"+ID+"_"
    print(dataset.get_no_classes())
    x3d.replace_logits(dataset.get_no_classes())
    if steps>0:
        load_ckpt = torch.load(model_save_path + '/x3d_PAR_rgb_ADAM_'+task+"_"+str(load_steps).zfill(6)+'.pt')
        x3d.load_state_dict(load_ckpt['model_state_dict'])

    x3d.cuda()
    x3d = nn.DataParallel(x3d)

    lr = init_lr
    print ('INIT LR: %f'%lr)


    # optimizer = optim.SGD(x3d.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    optimizer = optim.Adam(x3d.parameters(), lr=lr, weight_decay=1e-5)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)
    if steps > 0:
        optimizer.load_state_dict(load_ckpt['optimizer_state_dict'])
        lr_sched.load_state_dict(load_ckpt['scheduler_state_dict'])

    criterion = nn.BCEWithLogitsLoss()

    val_apm = APMeter()
    tr_apm = APMeter()
    best_map = 0

    while epochs < max_epochs:
        save_txt_path = save_txt_dir + "train_valid_stat_"+ID+ "_" + task +"_epoch_"+ str(epochs) + ".txt"
        epoch_time_remaining_path = save_txt_dir + task+"_"+ID+"_epoch_time_remaining.txt"
        with open(save_txt_path, 'w') as f:
            print ('Step {} Epoch {}'.format(steps, epochs))
            print ('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train']+['val']:
                # bar_st = iterations_per_epoch if phase == 'train' else val_iterations_per_epoch
                # bar = pkbar.Pbar(name='update: ', target=bar_st)
                if phase == 'train':
                    x3d.train(True)
                    epochs += 1
                    torch.autograd.set_grad_enabled(True)
                else:
                    x3d.train(False)  # Set model to evaluate mode
                    _ = x3d.module.aggregate_sub_bn_stats() # FOR EVAL AGGREGATE BN STATS
                    torch.autograd.set_grad_enabled(False)

                tot_loss = 0.0
                tot_cls_loss = 0.0
                num_iter = 0
                optimizer.zero_grad()

                # Iterate over data.
                print(phase)
                bar = tqdm.tqdm(enumerate(dataloaders[phase]),total=len(dataloaders))
                for i,data in bar:
                    num_iter += 1
                    with open(epoch_time_remaining_path, 'w') as progress_file:
                        progress_file.write(str(bar))
                    if phase == 'train':

                        inputs, labels = data

                        # print(inputs.shape)
                        # print(labels.shape)

                    else:
                        inputs, labels = data
                        b,n,c,t,h,w = inputs.shape # FOR MULTIPLE TEMPORAL CROPS
                        inputs = inputs.view(b*n,c,t,h,w)

                    inputs = inputs.cuda() # B 3 T W H
                    labels = labels.cuda() # B C

                    if phase == 'train':
                        logits, _, _ = x3d(inputs)
                        logits = logits.squeeze(2) # B C
                        probs = F.sigmoid(logits)
                        # print("Check output size")
                        # print(logits.shape)
                        # print(probs.shape)
                        #
                        # print("@" * 30)
                        # print(logits)
                        # print("@" * 30)

                    else:
                        with torch.no_grad():
                            logits, _, _ = x3d(inputs)
                        logits = logits.squeeze(2) # B C
                        logits = logits.view(b,n,logits.shape[1]) # FOR MULTIPLE TEMPORAL CROPS
                        probs = F.sigmoid(logits)
                        #probs = torch.mean(probs, 1)
                        #logits = torch.mean(logits, 1)
                        probs = torch.max(probs, dim=1)[0]
                        logits = torch.max(logits, dim=1)[0]


                    cls_loss = criterion(logits, labels)
                    tot_cls_loss += cls_loss.item()
                    bar.set_description("Tot Loss: {:.4f}".format(tot_cls_loss/(i+1)))
                    if phase == 'train':
                        tr_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())
                    else:
                        val_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())

                    loss = cls_loss/num_steps_per_update
                    tot_loss += loss.item()

                    if phase == 'train':
                        loss.backward()

                    # if num_iter == num_steps_per_update and phase == 'train':
                    if phase == 'train':
                        #lr_warmup(lr, steps-st_steps, warmup_steps, optimizer)
                        steps += 1
                        num_iter = 0
                        optimizer.step()
                        optimizer.zero_grad()
                        s_times = iterations_per_epoch//2
                        if (steps-load_steps) % s_times == 0:
                            tr_map = tr_apm.value().mean()
                            tr_apm.reset()
                            print (' Epoch:{} {} steps: {} Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}\n'.format(epochs, phase,
                                steps, tot_cls_loss/(s_times*num_steps_per_update), tot_loss/s_times, tr_map))

                            f.write (' Epoch:{} {} steps: {} Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}\n'.format(epochs, phase,
                                                                                                                steps,
                                                                                                                tot_cls_loss / (s_times * num_steps_per_update),
                                                                                                                tot_loss / s_times,
                                                                                                                tr_map))

                            tot_loss = tot_cls_loss = 0.


                if phase == 'val':
                    val_map = val_apm.value().mean()
                    lr_sched.step(tot_loss)
                    val_apm.reset()
                    print (' Epoch:{} {} Loc Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}\n'.format(epochs, phase,
                        tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, val_map))

                    f.write (' Epoch:{} {} Loc Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}\n'.format(epochs, phase,
                        tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, val_map))

                    tot_loss = tot_cls_loss = 0.
                    if val_map > best_map:
                        ckpt = {'model_state_dict': x3d.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': lr_sched.state_dict()}
                        best_map = val_map
                        best_epoch = epochs
                        # print (' Epoch:{} {} best mAP: {:.4f}\n'.format(best_epoch, phase, best_map))
                        f.write(' Epoch:{} {} best mAP: {:.4f}\n'.format(best_epoch, phase, best_map))
                        torch.save(ckpt, save_model+task+'_best.pt')

def lr_warmup(init_lr, cur_steps, warmup_steps, opt):
    start_after = 1
    if cur_steps < warmup_steps and cur_steps > start_after:
        lr_scale = min(1., float(cur_steps + 1) / warmup_steps)
        for pg in opt.param_groups:
            pg['lr'] = lr_scale * init_lr


def print_stats(long_ind, batch_size, stats, gamma_tau, bn_splits, lr):
    bs = batch_size * LONG_CYCLE[long_ind]
    if long_ind in [0,1]:
        bs = [bs*j for j in [2,1]]
        print(' ***** LR {} Frames {}/{} BS ({},{}) W/H ({},{}) BN_splits {} long_ind {} *****'.format(lr, stats[0][0], gamma_tau, bs[0], bs[1], stats[2][0], stats[3][0], bn_splits, long_ind))
    else:
        bs = [bs*j for j in [4,2,1]]
        print(' ***** LR {} Frames {}/{} BS ({},{},{}) W/H ({},{},{}) BN_splits {} long_ind {} *****'.format(lr, stats[0][0], gamma_tau, bs[0], bs[1], bs[2], stats[1][0], stats[2][0], stats[3][0], bn_splits, long_ind))


if __name__ == '__main__':
    run(task=args.task)
