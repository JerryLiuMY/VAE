import numpy as np
import matplotlib.pyplot as plt
from global_settings import device
from tools.utils import to_img
from params.params import params_dict
import torch
plt.ion()


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
