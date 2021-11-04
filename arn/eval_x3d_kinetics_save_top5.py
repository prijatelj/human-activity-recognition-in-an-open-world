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


from data.kinetics_eval import Kinetics as Kinetics_val
import numpy as np
import tqdm
from utils.apmeter import APMeter

import x3d as resnet_x3d

# from data.kinetics_multigrid import Kinetics_val

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
KINETICS_CLASS_LABELS = 'data/kinetics400_labels.txt'
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
ID = args.id
# set_batch_size
BS = 6
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
KINETICS_VAL_ROOT = lines[0][:-1]
KINETICS_VAL_ANNO = lines[1][:-1]
# KINETICS_VAL = lines[2][:-1]
model_save_path = lines[3][:-1]
save_txt_dir = lines[4][:-1]



# TODO: Why are these here? We can take the len() of the dataset
# TA2_DATASET_SIZE = {'train':13446, 'val':1491}
# TA2_DATASET_SIZE = {'train':4235, 'val':471} # UCF101 TA2 split
# TA2_DATASET_SIZE = {'train':1906, 'val':212} # HMDB51 TA2 split
# TA2_MEAN = [0, 0, 0]
# TA2_STD = [1, 1, 1]

# warmup_steps=0
def run(init_lr=INIT_LR, max_epochs=1, root=KINETICS_VAL_ROOT, anno=KINETICS_VAL_ANNO, val_anno=KINETICS_VAL_ANNO, batch_size=BS*BS_UPSCALE, task="class"):

    frames=80 # DOUBLED INSIDE DATASET, AS LONGER CLIPS
    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,256.], 'XL':[360.,450.]}[X3D_VERSION] # 'M':[256.,320.] FOR LONGER SCHEDULE
    gamma_tau = {'S':6, 'M':5, 'XL':5}[X3D_VERSION] # DOUBLED INSIDE DATASET, AS LONGER CLIPS

    st_steps = 0 # FOR LR WARM-UP
    load_steps = 0 # FOR LOADING AND PRINT SCHEDULE
    steps = 0
    epochs = 0
    KINETICS_MEAN = [110.63666788 / 255, 103.16065604 / 255, 96.29023126 / 255]
    KINETICS_STD = [38.7568578 / 255, 37.88248729 / 255, 40.02898126 / 255]

    validation_transforms = {
        'spatial':  Compose([CenterCropScaled(crop_size), #CenterCrop(crop_size),
                             ToTensor(255),
                             Normalize(KINETICS_MEAN, KINETICS_STD)]),
        'temporal': TemporalRandomCrop(frames, gamma_tau),
        'target':   ClassLabel()
    }

    validation_transforms = {
        'spatial':  Compose([CenterCropScaled(crop_size), #CenterCrop(crop_size),
                             ToTensor(255),
                             Normalize(KINETICS_MEAN, KINETICS_STD)]),
        'temporal': TemporalRandomCrop(frames, gamma_tau),
        'target':   ClassLabel()
    }


    val_dataset = Kinetics_val(
            root,
            KINETICS_VAL_ANNO,
            KINETICS_CLASS_LABELS,
            'val',
            spatial_transform = validation_transforms['spatial'],
            temporal_transform = validation_transforms['temporal'],
            target_transform = validation_transforms['target'],
            sample_duration=frames,
            gamma_tau=gamma_tau,
            crops=4)


    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)


    num_steps_per_update = 1 # ACCUMULATE GRADIENT IF NEEDED
    cur_iterations = steps * num_steps_per_update
    # iterations_per_epoch = len(dataset)//batch_size
    val_iterations_per_epoch = len(val_dataset)//(batch_size//2)
    # max_steps = iterations_per_epoch * max_epochs

    big_dumps = "/home/sgrieggs/big_dumps/" + root.split("/")[-2] + "/"

    try:
        os.makedirs(big_dumps)
    except FileExistsError:
        print("File Exists")

    dataloaders = {'val': val_dataloader}
    datasets = {'val': val_dataset}
    print('val',len(datasets['val']))
    # print('Total iterations:', max_steps, 'Total epochs:', max_epochs)
    print('datasets created')


    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=400, n_input_channels=3, dropout=0.5, base_bn_splits=1)

    # This is where we load the stuff

    load_ckpt = torch.load('models/x3d_multigrid_kinetics_fb_pretrained.pt')
    # load_ckpt = torch.load('/home/sgrieggs/Downloads/x3d_multigrid_kinetics_rgb_sgd_204000.pt')
    x3d.load_state_dict(load_ckpt['model_state_dict'])
    # save_model = model_save_path + '/x3d_PAR_rgb_ADAM_'+task+"_"+ID+"_"
    # print(dataset.get_no_classes())
    # x3d.replace_logits(dataset.get_no_classes())
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
        epochs += 1
        save_txt_path = save_txt_dir+ "/" + "train_valid_stat_"+ID+ "_" + task +"_epoch_"+ str(epochs) + ".txt"
        epoch_time_remaining_path = save_txt_dir + "/"+task+"_"+ID+"_epoch_time_remaining.txt"
        with open(save_txt_path, 'w') as f:
            print ('Step {} Epoch {}'.format(steps, epochs))
            print ('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['val']:
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
                bar = tqdm.tqdm(enumerate(dataloaders[phase]),total=len(dataloaders[phase]))
                right = 0
                for i,data in bar:
                    num_iter += 1
                    with open(epoch_time_remaining_path, 'w') as progress_file:
                        progress_file.write(str(bar))
                    if phase == 'train':

                        inputs, labels, path = data

                        # print(inputs.shape)
                        # print(labels.shape)

                    else:
                        inputs, labels, path = data
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
                            logits, feat, base = x3d(inputs)

                        # logits = logits.squeeze(2) # B C
                        # logits = logits.view(b,n,logits.shape[1]) # FOR MULTIPLE TEMPORAL CROPS
                        # probs = F.sigmoid(logits)
                        # #probs = torch.mean(probs, 1)
                        # #logits = torch.mean(logits, 1)
                        # probs = torch.sum(probs, dim=1)
                        # logits = torch.max(logits, dim=1)[0]
                        logits = logits.view(b, n, logits.shape[1])  # FOR MULTIPLE TEMPORAL CROPS
                        logits_sm = F.softmax(logits, dim=2)
                        logits_sm = torch.mean(logits_sm, 1)
                        logits = torch.mean(logits, 1)
                        for j, logit in enumerate(logits):
                            torch.save(logit, big_dumps + path[j].split('/')[-1].split('.')[0] + '.pt')
                            torch.save(labels[j], big_dumps + path[j].split('/')[-1].split('.')[0] + 'label.pt')
                            torch.save(feat[j], big_dumps + path[j].split('/')[-1].split('.')[0] + 'feat.pt')
                            # torch.save(base[j], big_dumps + path[j].split('/')[-1].split('.')[0] + 'base.pt')

                    for j, logit_sm in enumerate(logits_sm):
                        top5v, top5 = torch.topk(logit_sm, 5)
                        if torch.argmax(labels[j], dim=0).int() in top5:
                            right += 1

                    # right += torch.sum(torch.eq(torch.argmax(logits_sm, dim=1),torch.argmax(labels, dim=1)).int()).cpu().numpy()


                    cls_loss = criterion(logits, labels)
                    tot_cls_loss += cls_loss.item()
                    bar.set_description("Accuracy: {:.4f}".format(right/((i+1)*batch_size)))
                    if phase == 'train':
                        tr_apm.add(logits_sm.detach().cpu().numpy(), labels.cpu().numpy())
                    else:
                        val_apm.add(logits_sm.detach().cpu().numpy(), labels.cpu().numpy())

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
                        # s_times = iterations_per_epoch//2
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
                        # torch.save(ckpt, save_model+task+'_best.pt')

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
    #
#     targets = ["/media/sgrieggs/pageparsing/DATASETS/kinetics400_dataset/val_256_normal/",
# "/media/sgrieggs/pageparsing/DATASETS/kinetics400_dataset/val_256_Perspective/",
# "/media/sgrieggs/pageparsing/DATASETS/kinetics400_dataset/val_256_jitter/",
# "/media/sgrieggs/pageparsing/DATASETS/kinetics400_dataset/val_256_Noise/",
# "/media/sgrieggs/pageparsing/DATASETS/kinetics400_dataset/val_256_Blur/",
# "/media/sgrieggs/pageparsing/DATASETS/kinetics400_dataset/val_256_Invert/",
# "/media/sgrieggs/pageparsing/DATASETS/kinetics400_dataset/val_256_Rotation/",
# "/media/sgrieggs/pageparsing/DATASETS/kinetics400_dataset/val_256_Flip/"]

    # "/media/scratch_crc/kinetics-dataset-400-test-normal/",
    # "/media/scratch_crc/kinetics-dataset-400-test-blur/",
    # "/media/scratch_crc/kinetics-dataset-400-test-flip/",
    # "/media/scratch_crc/kinetics-dataset-400-test-invert/",
    targets = [
               "/media/sgrieggs/scratch365/kinetics-dataset-400-test-noise/",
               "/media/sgrieggs/scratch365/kinetics-dataset-400-test-perspective/",
               "/media/sgrieggs/scratch365/kinetics-dataset-400-test-rotation/",
               "/media/sgrieggs/scratch365/kinetics-dataset-400-test-jitter/"]

    for x in targets:
        run(root=x)
