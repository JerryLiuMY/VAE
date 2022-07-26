from datetime import datetime
from torch.nn import functional as F
from models.vae import VariationalAutoencoder
from params.params import train_dict
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_vae(data_loader, input_size):
    """ Training VAE with image dataset
    :param data_loader: image dataset loader
    :param input_size: size of input image
    :return:
    """

    # load parameters
    epoch, lr, beta = train_dict["epoch"], train_dict["lr"], train_dict["beta"]

    # building VAE
    vae = VariationalAutoencoder(input_size)
    vae = vae.to(device)
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Number of parameters: {num_params}")

    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)

    # set to training mode
    vae.train()
    for epoch in range(epoch):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training on epoch {epoch}")
        for img_batch, _ in data_loader:
            img_batch = img_batch.to(device)
            ima_batch_recon, mu, logvar = vae(img_batch)
            loss = vae_loss(ima_batch_recon, img_batch, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()


def vae_loss(recon_x, x, mu, logvar, beta):
    """ Calculating loss for variational autoencoder
    :param recon_x: reconstructed image
    :param x: original image
    :param mu: mean in the hidden layer
    :param logvar: log of the variance in the hidden layer
    :param beta: beta
    :return: loss function
    """

    # reconstruction loss (dependent of image resolution)
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction="sum")

    # KL-divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_div
