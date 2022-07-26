import torchvision.utils
import matplotlib.pyplot as plt
from global_settings import device
from tools.utils import to_img
from global_settings import OUTPUT_PATH
import numpy as np
import os
import torch
plt.ion()


def plot_output(model, images):
    """ Visualize original and reconstructed images
    :param model: trained vae model
    :param images: images
    :return:
    """

    # original images
    images_org = to_img(images)
    images_org = torchvision.utils.make_grid(images_org[0:50], 10, 5).numpy()
    images_org = np.transpose(images_org, (1, 2, 0))

    # reconstructed images
    model.eval()
    with torch.no_grad():
        images_rec = images.to(device)
        images_rec, _, _ = model(images_rec)
        images_rec = images_rec.cpu()
        images_rec = to_img(images_rec)
        images_rec = torchvision.utils.make_grid(images_rec[0:50], 10, 5).numpy()
        images_rec = np.transpose(images_rec, (1, 2, 0))

    # reconstructed images
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].imshow(images_org)
    axes[1].imshow(images_rec)
    visual_path = os.path.join(OUTPUT_PATH, "visual")
    fig.savefig(os.path.join(visual_path, "visual"))
