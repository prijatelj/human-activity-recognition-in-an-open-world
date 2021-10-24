from random import random

import cv2
import torch
import torchvision
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
from tqdm import tqdm
import numpy as np
import torchvision.io as io

import torch.nn.functional as F

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value


def my_video_loader(seq_path):
    frames = []
    # extract frames from the video
    if os.path.exists(seq_path):
        cap = cv2.VideoCapture(seq_path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert opencv image to PIL
            frames.append(frame)
    else:
        print('{} does not exist'.format(seq_path))

    return frames

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    torchvision.set_image_backend('accimage')
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'frame_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video

def get_video_container(path_to_vid, multi_thread_decode=False, backend="pyav"):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """
    if backend == "torchvision":
        with open(path_to_vid, "rb") as fp:
            container = fp.read()
        return container
    elif backend == "pyav":
        container = av.open(path_to_vid)
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        return container
    else:
        raise NotImplementedError("Unknown backend {}".format(backend))

def torchvision_decode(
    video_handle,
    sampling_rate,
    num_frames,
    clip_idx,
    video_meta,
    num_clips=10,
    target_fps=30,
    modalities=("visual",),
    max_spatial_scale=0,
    use_offset=False,
):
    """
    If video_meta is not empty, perform temporal selective decoding to sample a
    clip from the video with TorchVision decoder. If video_meta is empty, decode
    the entire video and update the video_meta.
    Args:
        video_handle (bytes): raw bytes of the video file.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the clip_idx-th video clip.
        video_meta (dict): a dict contains VideoMetaData. Details can be found
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        num_clips (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps.
        modalities (tuple): tuple of modalities to decode. Currently only
            support `visual`, planning to support `acoustic` soon.
        max_spatial_scale (int): the maximal resolution of the spatial shorter
            edge size during decoding.
    Returns:
        frames (tensor): decoded frames from the video.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): if True, the entire video was decoded.
    """
    # Convert the bytes to a tensor.
    video_tensor = torch.from_numpy(np.frombuffer(video_handle, dtype=np.uint8))

    decode_all_video = True
    video_start_pts, video_end_pts = 0, -1
    # The video_meta is empty, fetch the meta data from the raw video.
    if len(video_meta) == 0:
        # Tracking the meta info for selective decoding in the future.
        meta = io._probe_video_from_memory(video_tensor)
        # Using the information from video_meta to perform selective decoding.
        video_meta["video_timebase"] = meta.video_timebase
        video_meta["video_numerator"] = meta.video_timebase.numerator
        video_meta["video_denominator"] = meta.video_timebase.denominator
        video_meta["has_video"] = meta.has_video
        video_meta["video_duration"] = meta.video_duration
        video_meta["video_fps"] = meta.video_fps
        video_meta["audio_timebas"] = meta.audio_timebase
        video_meta["audio_numerator"] = meta.audio_timebase.numerator
        video_meta["audio_denominator"] = meta.audio_timebase.denominator
        video_meta["has_audio"] = meta.has_audio
        video_meta["audio_duration"] = meta.audio_duration
        video_meta["audio_sample_rate"] = meta.audio_sample_rate

    fps = video_meta["video_fps"]
    if (
        video_meta["has_video"]
        and video_meta["video_denominator"] > 0
        and video_meta["video_duration"] > 0
    ):
        # try selective decoding.
        decode_all_video = False
        clip_size = sampling_rate * num_frames / target_fps * fps
        start_idx, end_idx = get_start_end_idx(
            fps * video_meta["video_duration"],
            clip_size,
            clip_idx,
            num_clips,
            use_offset=use_offset,
        )
        # Convert frame index to pts.
        pts_per_frame = video_meta["video_denominator"] / fps
        video_start_pts = int(start_idx * pts_per_frame)
        video_end_pts = int(end_idx * pts_per_frame)

    # Decode the raw video with the tv decoder.
    v_frames, _ = io._read_video_from_memory(
        video_tensor,
        seek_frame_margin=1.0,
        read_video_stream="visual" in modalities,
        video_width=0,
        video_height=0,
        video_min_dimension=max_spatial_scale,
        video_pts_range=(video_start_pts, video_end_pts),
        video_timebase_numerator=video_meta["video_numerator"],
        video_timebase_denominator=video_meta["video_denominator"],
    )

    if v_frames.shape == torch.Size([0]):
        # failed selective decoding
        decode_all_video = True
        video_start_pts, video_end_pts = 0, -1
        v_frames, _ = io._read_video_from_memory(
            video_tensor,
            seek_frame_margin=1.0,
            read_video_stream="visual" in modalities,
            video_width=0,
            video_height=0,
            video_min_dimension=max_spatial_scale,
            video_pts_range=(video_start_pts, video_end_pts),
            video_timebase_numerator=video_meta["video_numerator"],
            video_timebase_denominator=video_meta["video_denominator"],
        )

    return v_frames, fps, decode_all_video

def get_start_end_idx(
    video_size, clip_size, clip_idx, num_clips, use_offset=False
):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips == 1:
                # Take the center clip if num_clips is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(delta / (num_clips - 1))
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def facebook_video_decoder(container,
    sampling_rate=5,
    num_frames=16,
    clip_idx=-1,
    num_clips=10,
    video_meta=None,
    target_fps=30,
    backend="trochvision",
    max_spatial_scale=256,
    use_offset=False,):
    """
    Shamelessly stolen from SlowFast
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        backend (str): decoding backend includes `pyav` and `torchvision`. The
            default one is `pyav`.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
    Returns:
        frames (tensor): decoded frames from the video.
    """
    # Currently support two decoders: 1) PyAV, and 2) TorchVision.
    # try:
        # if backend == "pyav":
        #     frames, fps, decode_all_video = pyav_decode(
        #         container,
        #         sampling_rate,
        #         num_frames,
        #         clip_idx,
        #         num_clips,
        #         target_fps,
        #         use_offset=use_offset,
        #     )
        # I'm not copying the code over for the other one lol
    try:
        if backend == "torchvision":
            frames, fps, decode_all_video = torchvision_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                video_meta,
                num_clips,
                target_fps,
                ("visual",),
                max_spatial_scale,
                use_offset=use_offset,
            )
        else:
            raise NotImplementedError
    except Exception as e:
        print("Failed to decode by {} with exception: {}".format(backend, e))
        return None

    # Return None if the frames was not decoded successfully.
    if frames is None or frames.size(0) == 0:
        return None

    clip_sz = sampling_rate * num_frames / target_fps * fps
    start_idx, end_idx = get_start_end_idx(
        frames.shape[0],
        clip_sz,
        clip_idx if decode_all_video else 0,
        num_clips if decode_all_video else 1,
        use_offset=use_offset,
    )
    # Perform temporal sampling from the decoded video.
    frames = temporal_sampling(frames, start_idx, end_idx, num_frames)
    return frames

def facebook_video_loader(seq_path,num_frames=16):
    video_container = None
    try:
        video_container = get_video_container(
            seq_path,
            True,
            "torchvision",
        )
    except Exception as e:
        print(e)

    sampling_rate = 5

    frames = facebook_video_decoder(
        video_container,
        sampling_rate,
        num_frames,
        temporal_sample_index,
        self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
        video_meta=self._video_meta[index],
        target_fps=30,
        backend="torchvision",
        max_spatial_scale=min_scale,
        use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
    )
    return frames


def get_default_video_loader():
    # image_loader = get_default_image_loader()
    return functools.partial(my_video_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    data = open(data).read().splitlines()
    for class_label in data: #['labels']
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

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

    return video_names, annotations


def make_dataset(root_path, annotation_path, class_labels, subset, n_samples_for_each_video, sample_duration):
    print("----------------------------------------------------")
    print(root_path)
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
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
                'video_id': video_names[i].split('/')[1].split('.mp4')[0]
            }

            if len(annotations) != 0:
                sample['label'] = class_to_idx[annotations[i]['label']]
            else:
                sample['label'] = -1
            num_frames = 0
            # cap = cv2.VideoCapture(video_path)
            # while (cap.isOpened()):
            #     ret, frame = cap.read()
            #     if ret == False:
            #         break
            #     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert opencv image to PIL
            #     # frames.append(frame)
            #     num_frames += 1

            # num_frames = int(data['database'][video_names[i]]['annotations']['segment'][1])
            # if num_frames > 0:
            #     num_frames = max(2 * 80 + 2, num_frames)
            # else:
            #     continue
            #
            # label = np.zeros((len(), num_frames), np.float32)
            # cur_class_idx = class_to_idx[annotations[i]]
            # label[cur_class_idx, :] = 1
            # dataset.append((video_path, label, num_frames))
            dataset.append(sample)
        # np.save(pre_saved_dataset, dataset)

    return dataset, idx_to_class


class Kinetics(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 class_labels,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 gamma_tau=5,
                 crops=10,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, class_labels, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.gamma_tau = gamma_tau
        self.crops = crops
        self.sample_duration = sample_duration
        self.frames = sample_duration//gamma_tau


    def __getitem__(self, index):
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

        # if step == 0:
        clips = [clip[:,:self.frames,...] for i in range(self.crops)]
        clips = torch.stack(clips, 0)
        # else:
        # clips = [clip[:,i:i+self.frames,...] for i in range(0, step*self.crops, step)]
        # clips = torch.stack(clips, 0)
        target = F.one_hot(torch.tensor(self.data[index]['label']), num_classes=len(self.class_names)).float()
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        if clips.shape[2] != 16:
            print(path)
        return clips, target

    def __len__(self):
        return len(self.data)
