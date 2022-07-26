import numpy as np
import matplotlib.pyplot as plt
from global_settings import device
from tools.utils import to_img
from params.params import params_dict
import torch
plt.ion()


def sample(model):
    model.eval()
    with torch.no_grad():
        # sample latent from normal
        hidden = params_dict["hidden"]
        latent = torch.randn(128, hidden, device=device)

        # build reconstruction
        image_sample = model.decoder(latent).cpu()

        return image_sample


def show_space(model):

    hidden = params_dict["hidden"]
    if hidden != 2:
        raise ValueError("Please change the parameters to two latent dimensions!")

    model.eval()
    with torch.no_grad():
        # create grid in 2d latent space
        latent_x = np.linspace(-1.5, 1.5, 20)
        latent_y = np.linspace(-1.5, 1.5, 20)
        latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)
        for i, lx in enumerate(latent_x):
            for j, ly in enumerate(latent_y):
                latents[j, i, 0] = lx
                latents[j, i, 1] = ly
        latents = latents.view(-1, 2)

        #  build reconstruction
        latents = latents.to(device)
        image_space = model.decoder(latents).cpu()

        return image_space


def interpolation(model, lbd, image_1, image_2):
    model.eval()
    with torch.no_grad():
        # reconstruct latent vector
        image_1 = image_1.to(device)
        image_2 = image_2.to(device)
        latent_1, _ = model.encoder(image_1)
        latent_2, _ = model.encoder(image_2)

        # interpolate and build reconstruction
        latent_inter = lbd * latent_1 + (1 - lbd) * latent_2
        image_inter = model.decoder(latent_inter).cpu()

        return image_inter


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
