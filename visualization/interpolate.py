import numpy as np
import matplotlib.pyplot as plt
from global_settings import device
from tools.utils import to_img
import torch
plt.ion()


def plot_interpolation(model, valid_loader):
    # sort part of test set by digit
    digits = [[] for _ in range(10)]
    for valid_batch, label_batch in valid_loader:
        for i in range(valid_batch.size(0)):
            digits[label_batch[i]].append(valid_batch[i:i + 1])
        if sum(len(d) for d in digits) >= 1000:
            break

    # interpolation lambdas
    lambda_lin = np.linspace(0, 1, 10)
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()

    for idx, l in enumerate(lambda_lin):
        inter_image = interpolation(model, float(l), digits[7][0], digits[1][0])
        inter_image = to_img(inter_image)
        axs[idx].imshow(inter_image[0, 0, :, :], cmap="gray")
        axs[idx].set_title("lambda_val=" + str(round(l, 1)))


def interpolation(model, lambda1, image_1, image_2):
    model.eval()
    with torch.no_grad():
        # reconstruct latent vector
        image_1 = image_1.to(device)
        image_2 = image_2.to(device)
        latent_1, _ = model.encoder(image_1)
        latent_2, _ = model.encoder(image_2)

        # interpolate and reconstruct image
        inter_latent = lambda1 * latent_1 + (1 - lambda1) * latent_2
        inter_image = model.decoder(inter_latent).cpu().numpy()

        return inter_image

