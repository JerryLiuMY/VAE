import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
from global_settings import device
from tools.utils import to_img
import torch
plt.ion()


def visualise_output(model, images):
    """ Visualize original and reconstructed images
    :param model:
    :param images:
    :return:
    """

    # original images
    show_image(torchvision.utils.make_grid(images[1:50], 10, 5))
    plt.show()

    # reconstructed images
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        images, _, _ = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()


def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
