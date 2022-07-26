import torchvision.utils
import matplotlib.pyplot as plt
from global_settings import device
from params.params import params_dict
from tools.utils import to_img
from global_settings import OUTPUT_PATH
import numpy as np
import os
import torch
plt.ion()


def plot_vae(model, image):
    """ Visualize original and reconstructed images
    :param model: trained vae model
    :param image: input image
    """

    # original images
    image_org = to_img(image)
    image_org = torchvision.utils.make_grid(image_org[0:50], 10, 5).numpy()
    image_org = np.transpose(image_org, (1, 2, 0))

    # reconstructed images
    model.eval()
    with torch.no_grad():
        image_rec = image.to(device)
        image_rec, _, _ = model(image_rec)
        image_rec = image_rec.cpu()
        image_rec = to_img(image_rec)
        image_rec = torchvision.utils.make_grid(image_rec[0:50], 10, 5).numpy()
        image_rec = np.transpose(image_rec, (1, 2, 0))

    # reconstructed images
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].imshow(image_org)
    axes[1].imshow(image_rec)
    visual_path = os.path.join(OUTPUT_PATH, "visual")
    fig.savefig(os.path.join(visual_path, "recon.pdf"), bbox_inches="tight")


def plot_sample(model):
    """ Visualize sampled images
    :param model: trained vae model
    """

    model.eval()
    with torch.no_grad():
        # sample latent from normal
        hidden = params_dict["hidden"]
        latent = torch.randn(128, hidden, device=device)

        # build reconstruction
        image_smp = model.decoder(latent).cpu()
        image_smp = torchvision.utils.make_grid(image_smp.data[:100], 10, 5).numpy()
        image_smp = np.transpose(image_smp, (1, 2, 0))

    # reconstructed images
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_smp)
    visual_path = os.path.join(OUTPUT_PATH, "visual")
    fig.savefig(os.path.join(visual_path, "sample.pdf"), bbox_inches="tight")


def plot_interpolation(model, digits):
    """ Visualize interpolated images
    :param model: trained vae model
    :param digits: image tensors grouped by digits label
    """

    # interpolation lambdas
    lambda_lin = np.linspace(0, 1, 10)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axes = axes.ravel()

    for idx, lbd in enumerate(lambda_lin):
        inter_image = interpolation(model, float(lbd), digits[7][0], digits[1][0])
        inter_image = to_img(inter_image)
        axes[idx].imshow(inter_image[0, 0, :, :], cmap="gray")
        axes[idx].set_title("lambda_val=" + str(round(lbd, 1)))

    visual_path = os.path.join(OUTPUT_PATH, "visual")
    fig.savefig(os.path.join(visual_path, "interpolation.pdf"), bbox_inches="tight")


def interpolation(model, lbd, img_1, img_2):
    """ Perform interpolation of two images
    :param model: trained vae model
    :param lbd: lambda value
    :param img_1: first image
    :param img_2: second image
    :return: interpolate images
    """

    model.eval()
    with torch.no_grad():
        # reconstruct latent vector
        img_1 = img_1.to(device)
        img_2 = img_2.to(device)
        mu_1, _ = model.encoder(img_1)
        mu_2, _ = model.encoder(img_2)

        # interpolate and build reconstruction
        mu_inter = lbd * mu_1 + (1 - lbd) * mu_2
        image_inter = model.decoder(mu_inter).cpu()

        return image_inter
