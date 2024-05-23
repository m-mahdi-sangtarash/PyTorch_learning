import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

plt.rcParams["savefig.bbox"] = 'tight'

def visualize(imgs, **imshow_kwargs):
    fig, axs = plt.subplots(1, len(imgs), figsize=(15, 15))
    for row_idx, row in enumerate([imgs]):
        for col_idx, img in enumerate(row):
            ax = axs[col_idx]
            ax.imshow(np.asarray(img))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()


img_path = r"./Img/tomato.jpg"
original_img = Image.open(img_path)

# Transforms

# Pad
padded_imgs = [transforms.Pad(padding=padding, fill=125, padding_mode='reflect')(original_img) for padding in (5, 15, 20, 45)]
visualize(padded_imgs)

# Resize
resize_imgs = [transforms.Resize(size=size) (original_img) for size in (30, 50 , 100, original_img.size)]
visualize(resize_imgs)

# CenterCrop
centerCrop_imgs = [transforms.CenterCrop(size=size) (original_img) for size in (30, 50 , 100, original_img.size)]
visualize(centerCrop_imgs)

# FiveCrop
(top_left, top_right, bottom_left, bottom_right, center) = transforms.FiveCrop(size=(100, 100)) (original_img)
visualize([top_left, top_right, bottom_left, bottom_right, center])

# GrayScale
grayScale_imgs = transforms.Grayscale()(original_img)
