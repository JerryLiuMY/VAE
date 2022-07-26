import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from global_settings import OUTPUT_PATH, device


def plot_inter(model, digit_set, d1, d2):
    """ Visualize interpolated images
    :param model: trained vae model
    :param digit_set: image tensors grouped by digits label
    :param d1: first digit selected for interpolation
    :param d2: second digit selected for interpolation
    """

    # interpolation lambdas
    lambda_lin = np.linspace(0, 1, 10)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axes = axes.ravel()

    for idx, lbd in enumerate(lambda_lin):
        inter_image = interpolation(model, float(lbd), digit_set[d1][0], digit_set[d2][0])
        inter_image = inter_image.clamp(0, 1)
        axes[idx].imshow(inter_image[0, 0, :, :], cmap="gray")
        axes[idx].set_title("lambda_val=" + str(round(lbd, 1)))

    visual_path = os.path.join(OUTPUT_PATH, "visualizations")
    fig.savefig(os.path.join(visual_path, "inter.pdf"), bbox_inches="tight")


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
