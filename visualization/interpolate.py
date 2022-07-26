import numpy as np
import matplotlib.pyplot as plt
from global_settings import device
from tools.utils import to_img
import torch
plt.ion()


def plot_interpolation(model, digits):
    # interpolation lambdas
    lambda_lin = np.linspace(0, 1, 10)
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()

    for idx, lbd in enumerate(lambda_lin):
        inter_image = interpolation(model, float(lbd), digits[7][0], digits[1][0])
        inter_image = to_img(inter_image)
        axs[idx].imshow(inter_image[0, 0, :, :], cmap="gray")
        axs[idx].set_title("lambda_val=" + str(round(lbd, 1)))


def interpolation(model, lbd, image_1, image_2):
    model.eval()
    with torch.no_grad():
        # reconstruct latent vector
        image_1 = image_1.to(device)
        image_2 = image_2.to(device)
        latent_1, _ = model.encoder(image_1)
        latent_2, _ = model.encoder(image_2)

        # interpolate and reconstruct image
        inter_latent = lbd * latent_1 + (1 - lbd) * latent_2
        inter_image = model.decoder(inter_latent).cpu().numpy()

        return inter_image

