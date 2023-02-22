import os
import numpy as np
import torch
import torchvision.utils
from matplotlib import pyplot as plt
from global_settings import device, OUTPUT_PATH
from params.params import params_dict


def plot_space(model):
    """ Visualize sampled images
    :param model: trained vae model
    """

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
        image_spc = model.decoder(latents).cpu()

    image_spc = torchvision.utils.make_grid(image_spc.data[:100], 10, 5).numpy()
    image_spc = np.transpose(image_spc, (1, 2, 0))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_spc)
    visual_path = os.path.join(OUTPUT_PATH, "visualizations")
    fig.savefig(os.path.join(visual_path, "space.pdf"), bbox_inches="tight")
