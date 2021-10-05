import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pylab
from augmentation import *


orig_img = cv2.cvtColor(cv2.imread("/home/sgrieggs/Downloads/Lenna.png"), cv2.COLOR_BGR2RGB)
def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            if isinstance(img,torch.Tensor):
                img = img.permute(1, 2, 0)
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()
# for x in range(3):
#     torch.manual_seed(0)
#     for y in range(3):
#         img = cv2.cvtColor(cv2.imread("/home/sgrieggs/Downloads/Lenna.png"), cv2.COLOR_BGR2RGB)
#         img = torch.tensor(img).permute(2, 0, 1)
#         perspective_transformer = T.ColorJitter(brightness=.5, hue=.3)
#         # perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)
#         perspective_imgs = [perspective_transformer(img) for _ in range(4)]
#         perspective_imgs2 = []
#         # for x in range(4):
#         #     s,e = perspective_transformer.get_params(img.shape[1], img.shape[2])
#         #     perspective_imgs2.append(F.perspective(img,s,e))
#         plot(perspective_imgs)
#         # plot(perspective_imgs2)
#     print("-------------------------------------")

img = cv2.cvtColor(cv2.imread("/home/sgrieggs/Downloads/Lenna.png"), cv2.COLOR_BGR2RGB)
pylab.imshow(img)
pylab.show()

test = Rotation(degrees=(0, 180), fill=-1)
img1,params = test.augment(img)
img2 = test.Rotation(torch.tensor(img).permute(2, 0, 1),params).permute(1,2,0).numpy()


print(np.equal(img1,img2).all())
pylab.imshow(img1)
pylab.show()
pylab.imshow(img2)
pylab.show()


