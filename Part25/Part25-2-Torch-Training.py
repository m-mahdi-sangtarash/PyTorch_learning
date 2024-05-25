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

# Auto Augmentations

# RandAugment Transform

randAug_imgs = [transforms.RandAugment()(original_img) for _ in range(4)]
visualize(randAug_imgs)


# AugMix Transform
augmix_imgs = [transforms.AugMix()(original_img) for _ in range(4)]
visualize(augmix_imgs)
