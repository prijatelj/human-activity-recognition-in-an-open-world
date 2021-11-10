"""Consolidating the reused data functions."""
from enum import Enum
import os
import functools
import logging

import cv2
import numpy as np
import torchvision
from PIL import Image


class VideoStatus(Enum):
    """The different post-loading status of videos."""
    LOADED = 0  # 'loaded'
    MISSING = 1 # 'missing'
    CORRUPT = 2 # 'bad_video'


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
            # convert opencv image to PIL
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    else:
        print('{} does not exist'.format(seq_path))
    if len(frames) == 0:
        print(seq_path + " is busted")
        frames = [np.zeros((360, 640, 3)).astype(np.uint8)]
    return frames


def status_video_frame_loader(path):
    """Loads the frames of the video, logs video errors, returns status.
    Args
    ----
    path : str
        The filepath to the video whose frames are to be loaded.

    Returns
    -------
    (list(np.array), VideoStatus)
        The list of video frames as images and a string representing the status
        of the associated video frames as 'path_dne' for when the filepath does
        not exist, 'bad_video' for when the video loaded improperly, and
        'loaded' for when the video frames loaded as expected.
    """
    frames = []

    # Extract the frames from the video
    if os.path.exists(path): # NOTE may have to worry about >1 batch sizes.
        cap = cv2.VideoCapture(path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            # convert opencv image to PIL
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    else:
        logging.warning('%s does not exist', path)
        return [np.zeros((360, 640, 3)).astype(np.uint8)], VideoStatus.MISSING

    if len(frames) == 0:
        logging.warning("%s is busted", path)
        return [np.zeros((360, 640, 3)).astype(np.uint8)], VideoStatus.CORRUPT

    return frames, VideoStatus.LOADED


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


def get_default_video_loader():
    # image_loader = get_default_image_loader()
    return functools.partial(my_video_loader)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    data = open(data).read().splitlines()
    for class_label in data: #['labels']
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map
