import os
import cv2
import numpy as np

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
    print(cv2.VideoWriter_fourcc('m', 'p', '4', 'v'))
    width, height, fps, codec = params
    print(codec)
    size = (width, height)
    result = cv2.VideoWriter('/home/sgrieggs/Image/filename.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
    for frame in frames:
        result.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


# /home/sgrieggs/Image/
frames, params = my_video_loader("/media/sgrieggs/pageparsing/Kinetics-700/val/200/AuMfvvCk_2A.mp4")
my_video_saver('/home/sgrieggs/Image/filename.mp4',frames, params)
