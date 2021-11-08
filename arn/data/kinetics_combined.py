"""Kinetics dataloader for the unified Kinetics labels CSV."""
import os

import pandas as pd
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


def load_annotation_data(data_file_path, subset):
    df = pd.read_csv(data_file_path)
    if isinstance(subset, int):
        index = len(df.index) // 50
        if subset == 50:
            df = df[index * (subset-1):]
        else:
            df = df[index * (subset-1): index*subset]
    return df.where(df.notnull(), None)


def get_video_names_and_annotations(
    data,
    root="/media/scratch_crc/dprijate/osr/har/data/kinetics/",
):
    video_names = []
    annotations = []
    bar = tqdm(data.iterrows(), total=len(data.index))
    # bad  = 0
    # bad_list = []
    for i, row in bar:
        # print(row)
        id = row['youtube_id']
        start = row['time_start']
        end = row['time_end']
        video_name = '{}_{:06d}_{:06d}.mp4'.format(id, int(start), int(end))
        k400s = row['split_kinetics400']
        k600s = row['split_kinetics600']
        k700s = row['split_kinetics700_2020']
        if k400s is not None:
            if k400s == 'train':
                target = "/media/scratch_crc/sgrieggs/kinetics-dataset-400-train/"+video_name
            elif k400s == 'validate':
                target = "/media/scratch_crc/sgrieggs/kinetics-dataset-400-val/"+video_name
            elif k400s == 'test':
                target = "/media/scratch_crc/sgrieggs/kinetics-dataset-400-test/"+video_name
            else:
                print(k400s)
                print("BAD! c")
        elif k600s is not None:
            if k600s == 'train':
                target = root + "kinetics600/videos/train/"+video_name
            elif k600s == 'validate':
                target = root + "kinetics600/videos/validate/"+video_name
            elif k600s == 'test':
                target = root + "kinetics600/videos/test/"+video_name
            else:
                print(k600s)
                print("BAD! b")
        elif k700s is not None:
            if k700s == 'train':
                target = root + "kinetics700_2020/videos/train/" + row['label_kinetics700_2020'] +'/' + video_name
            elif k700s == 'validate':
                target = root + "kinetics700_2020/videos/validate/"+ row['label_kinetics700_2020'] +'/' +video_name
            elif k700s == 'test':
                target = root + "kinetics700_2020/videos/test/"+video_name
            else:
                print(k700s)
                print("BAD! a")
        else:
            print(row)
        video_names.append(target)
        annotations.append({"id":'{}_{:06d}_{:06d}'.format(id, int(start), int(end)),"segment":(start,end)})

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset=50, n_samples_for_each_video=1, sample_duration=10,outpath="/scratch365/sgrieggs/humongous_big_dumps"):

    data = load_annotation_data(annotation_path,subset)
    video_names, annotations = get_video_names_and_annotations(data, root_path)

    dataset = []
    for i in tqdm(range(len(video_names))):
        video_path = video_names[i]
        if not os.path.exists(video_path):
            continue
        if os.path.exists(outpath+annotations[i]['id']+"_logits.pt"):
            if os.path.exists(outpath+annotations[i]['id']+"_feat.pt"):
                print(outpath+annotations[i]['id']+"_feat.pt and " + outpath+annotations[i]['id']+"_logits.pt are already there")
                continue

        sample = {
            'video': video_path,
            'segment': annotations[i]['segment'],
        }
        # 'video_id': video_names[i].split('/')[1].split('.mp4')[0]
        sample['label'] = annotations[i]['id']
        num_frames = 0

        dataset.append(sample)

    return dataset


class Kinetics(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional):
            A function/transform that takes in an PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional):
            A function/transform that takes in a list of frame indices and
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
        outpath="/home/sgrieggs/bigger_dumps/",
        randomize_spatial_params=True,
    ):
        self.data = make_dataset(
            root_path,
            annotation_path,
            subset,
            n_samples_for_each_video,
            sample_duration,
            outpath,
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
        target = self.data[index]['label']
        # print(clips.shape)
        return clips, target, path


    def __len__(self):
        return len(self.data)
