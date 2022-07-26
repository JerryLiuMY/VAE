import os
import numpy as np
import torch
import torchvision.utils
from matplotlib import pyplot as plt
from global_settings import device, OUTPUT_PATH
from params.params import params_dict


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
