from multiprocessing import Pool
import os
import cv2
import numpy as np
import glob
import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
from models.augmentation import *



def my_video_loader(seq_path):
    frames = []
    # extract frames from the video
    if os.path.exists(seq_path):
        cap = cv2.VideoCapture(seq_path)
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(5))
        codec = cap.get(6)
        cnt = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert opencv image to PIL
            frames.append(frame)
        #print(seq_path, len(frames), min(frame_indices), max(frame_indices), test)
    else:
        print('{} does not exist'.format(seq_path))
    return frames, (int(width),int(height),int(fps),int(codec))

def my_video_saver(seq_path,frames,params):
    print(seq_path)
    # print(cv2.VideoWriter_fourcc('m', 'p', '4', 'v'))
    width, height, fps, codec = params
    # print(codec)
    size = (width, height)
    result = cv2.VideoWriter(seq_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
    for frame in frames:
        result.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def augment_all(frame_list, aug_func, same=True):
    if same:
        frame, params = aug_func.augment(frame_list[0])
        frames = [frame]
        for x in range(1,len(frame_list)):
            frames.append(aug_func.aug(torch.tensor(frame_list[x]).permute(2, 0, 1),params).permute(1,2,0).numpy())
    else:
        frames = []
        for frame in frame_list:
            frames.append(aug_func.augment(frame)[0])
    return frames

def process_targets(target):
    output_dir = "/media/sgrieggs/pageparsing/DATASETS/kinetics400_dataset/val_256_Perspective/"
    test = target.split("/")[-2:]
    try:
        os.makedirs(output_dir+test[0])
    except FileExistsError as e:
        exists = True
        # print("test")
    frames, params = my_video_loader(target)
    frames = augment_all(frames, PerspectiveTransform(), same=True)
    my_video_saver(output_dir+"/".join(test),frames, params)
    return True

# class ColorJitter(StochasticAugmenter):
# class Noise(StochasticAugmenter):
# class Blur(StochasticAugmenter):
# class InvertColor(Augmenter):
# class PerspectiveTransform(StochasticAugmenter):
# class Rotation(StochasticAugmenter):
# class Flip(Augmenter):


# /home/sgrieggs/Image/
# frames, params = my_video_loader("/media/sgrieggs/pageparsing/Kinetics-700/val/200/AuMfvvCk_2A.mp4")

# my_video_saver('/home/sgrieggs/Image/filename.mp4',frames, params)
targets = glob.glob("/media/sgrieggs/pageparsing/DATASETS/kinetics400_dataset/val_256/*/*.mp4")
out_file_names = []
# process_targets(targets[0])

r = process_map(process_targets, targets[:30], max_workers=30)
