import torch
from torch.nn import functional as F


def elbo_binary(x, recon, mu, logvar, beta):
    """ Calculating binary loss for variational autoencoder
    :param recon: reconstructed image
    :param x: original image
    :param mu: mean in the hidden layer
    :param logvar: log of the variance in the hidden layer
    :param beta: beta
    :return: reconstruction loss + KL
    """

    # KL-divergence
    kl_div = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # reconstruction loss (dependent of image resolution)
    recon_loss = - F.binary_cross_entropy(recon.view(-1, 784), x.view(-1, 784), reduction="sum")

    # define loss
    loss = - beta * kl_div + recon_loss

    return - loss


def elbo_l2(x, recon, mu, logvar, beta):
    """ Calculating binary loss for variational autoencoder
    :param recon: reconstructed image
    :param x: original image
    :param mu: mean in the hidden layer
    :param logvar: log of the variance in the hidden layer
    :param beta: beta
    :return: reconstruction loss + KL
    """

    # KL-divergence
    kl_div = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # reconstruction loss (dependent of image resolution)
    recon_loss = F.mse_loss(recon.view(-1, 784), x.view(-1, 784))

    # define loss
    loss = - beta * kl_div + recon_loss

    return - loss
