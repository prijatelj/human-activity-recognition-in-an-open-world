"""Kinetics Dataloader for a single Kinetics dataset, 400 or 600. Not 700."""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm

from arn.data.dataloder_utils import (
    load_value_file,
    my_video_loader,
    pil_loader,
    get_default_image_loader,
    video_loader,
    get_default_video_loader,
    get_class_labels,
)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_video_names_and_annotations(data, subset, source="torrent"):
    video_names = []
    annotations = []
    if source == "torrent":
        for key, value in data.items():
            this_subset = value['subset']
            # if this_subset == subset:
            start_and_end_times = {}
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            elif subset == 'train':
                st = int(value['annotations']['segment'][0])
                end = int(value['annotations']['segment'][1])
                label = value['annotations']['label'].replace(' ','_')
                video_names.append('{}/{}_{}_{}'.format(label, key, str(st).zfill(6), str(end).zfill(6)))
                annotations.append(value['annotations'])
            else:
                label = value['annotations']['label'].replace(' ','_').replace('(','-').replace(')','-').replace("'",'-')
                video_names.append('{}/{}.mp4'.format(label, key.lstrip('-')))
                annotations.append(value['annotations'])
    elif source == "cdvf":
        for key, value in data.items():
            this_subset = value['subset']
            # if this_subset == subset:
            start_and_end_times = {}
            if subset == 'testing':
                label = value['annotations']['segment']
                video_names.append('{}_{:06d}_{:06d}.mp4'.format(key,int(label[0]),int(label[1])))
                annotations.append(value['annotations'])
                # video_names.append('test/{}'.format(key))
            elif subset == 'train':
                st = int(value['annotations']['segment'][0])
                end = int(value['annotations']['segment'][1])
                label = value['annotations']['label'].replace(' ','_')
                video_names.append('{}/{}_{}_{}'.format(label, key, str(st).zfill(6), str(end).zfill(6)))
                annotations.append(value['annotations'])
            else:
                label = value['annotations']['segment']
                video_names.append('{}_{:06d}_{:06d}.mp4'.format(key,int(label[0]),int(label[1])))
                annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, class_labels, subset, n_samples_for_each_video, sample_duration):
    print("----------------------------------------------------")
    print(root_path)
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset,"cdvf")
    class_to_idx = get_class_labels(class_labels)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    pre_saved_dataset = os.path.join(root_path, 'labeldata_80.npy')
    if os.path.exists(pre_saved_dataset):
        print('{} exists'.format(pre_saved_dataset))
        dataset = np.load(pre_saved_dataset, allow_pickle=True)
    else:
        dataset = []
        for i in tqdm(range(len(video_names))):
            video_path = root_path + video_names[i]

            if not os.path.exists(video_path):
                continue

            sample = {
                'video': video_path,
                'segment': annotations[i]['segment'],
            }
            # 'video_id': video_names[i].split('/')[1].split('.mp4')[0]
            if len(annotations) != 0:
                sample['label'] = class_to_idx[annotations[i]['label']]
            else:
                sample['label'] = -1
            num_frames = 0
            dataset.append(sample)

    return dataset, idx_to_class


class Kinetics(data.Dataset):
    """General Kinetics Torch Dataset. Able to load Kinetics 400, 600, or 700.
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional):
            A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional):
            A function/transform that  takes in a list of frame indices and
            returns a transformed version
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
        loader (callable, optional):
            A function to load an video given its path and frame indices.

    Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(
        self,
        root_path,
        annotation_path,
        class_labels,
        subset,
        n_samples_for_each_video=10,
        spatial_transform=None,
        temporal_transform=None,
        target_transform=None,
        sample_duration=16,
        gamma_tau=5,
        crops=10,
        get_loader=get_default_video_loader,
        randomize_spatial_params=True,
    ):
        self.data, self.class_names = make_dataset(
            root_path,
            annotation_path,
            class_labels,
            subset,
            n_samples_for_each_video,
            sample_duration,
        )
        self.subset = subset
        self.samples = n_samples_for_each_video
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.gamma_tau = gamma_tau
        self.crops = crops
        self.sample_duration = sample_duration
        self.frames = sample_duration//gamma_tau

        self.randomize_spatial_params = randomize_spatial_params

    def old__getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        clip = self.loader(path)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(Image.fromarray(img)) for img in clip]
        else:
            clip = [Image.fromarray(img) for img in clip]

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3) # T C H W --> C T H W

        # # if step == 0:
        # clips = [clip[:,:self.frames,...] for i in range(self.crops)]
        # clips = torch.stack(clips, 0)
        # else:
        clips = [clip[:,i:i+self.frames,...] for i in range(0, step*self.crops, step)]
        clips = torch.stack(clips, 0)
        target = F.one_hot(torch.tensor(self.data[index]['label']), num_classes=len(self.class_names)).float()
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        # if clips.shape[2] != 16:
        #     print(path)
        return clips, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
            TODO specify image type and expected general shape
        """
        path = self.data[index]['video']
        clip = self.loader(path)
        frame_indices = list(range(0, len(clip)))
        #if self.temporal_transform is not None:
        #    frame_indices = self.temporal_transform(frame_indices)

        # FOR MULTI-CROP TESTING
        frame_indices = frame_indices[::self.gamma_tau]
        clip = [clip[i] for i in frame_indices]
        step = int((len(frame_indices) -  self.frames)//(self.crops))

        # clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            if self.randomize_spatial_params:
                self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(Image.fromarray(img)) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3) # T C H W --> C T H W

        if step == 0:
            clips = [clip[:,:self.frames,...] for i in range(self.crops)]
            clips = torch.stack(clips, 0)
        else:
            clips = [clip[:,i:i+self.frames,...] for i in range(0, step*self.crops, step)]
            for i in range(len(clips)):
                clp = clips[i]
                if clp.shape[1] != self.frames:
                    # if self.frames-clp.shape[1] != 1:
                        # print("interesting... " +str(self.frames-clp.shape[1]) )
                    # Padding in torch is absolutely bonkers, lol this pads dimension 1
                    p2d = (0, 0, 0, 0, 0, self.frames-clp.shape[1])
                    clips[i] = F.pad(clp, p2d, "constant", 0)

            clips = torch.stack(clips, 0)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = F.one_hot(torch.tensor(self.data[index]['label']), num_classes=len(self.class_names)).float()

        # print(clips.shape)
        return clips, target

    def __len__(self):
        return len(self.data)
