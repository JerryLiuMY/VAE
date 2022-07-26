import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
from global_settings import device
import torch
plt.ion()


def visualise_output(model, images):

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


def to_img(x):
    """ This function takes as an input the
    :param x: reconstructed image
    :return:
    """
    x = x.clamp(0, 1)
    return x


def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
