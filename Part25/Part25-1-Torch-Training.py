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

# ColorJitter Transform
jitter_transform = transforms.ColorJitter(brightness=0.5,
                                          contrast=0.5,
                                          saturation=0.5,
                                          hue=0.5)
jitted_img = [jitter_transform(original_img) for _ in range(4)]
visualize(jitted_img)

# RandomPerspective Transform
rand_persp = transforms.RandomPerspective(distortion_scale=0.5,
                                          p=1.0)
perspective_imgs = [rand_persp(original_img) for _ in range(4)]

visualize(perspective_imgs)

# RandomRotation Transform
rotated_imgs = [transforms.RandomRotation(degrees=(0, 90))(original_img) for _ in range(4)]
visualize(rotated_imgs)


# RandomCrop Transform
rand_cropped_imgs = [transforms.RandomCrop(size=(100, 100))(original_img) for _ in range(4)]
visualize(rand_cropped_imgs)

# RandomVerticalFlip Transfom
vflipped_imgs = [transforms.RandomVerticalFlip(p=0.5)(original_img) for _ in range(4)]
visualize(vflipped_imgs)

