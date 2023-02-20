from models.elbo import elbo_binary, elbo_l2
from models.vae import VariationalAutoencoder
from params.params import train_dict
from global_settings import device
from datetime import datetime
import torch
import numpy as np


def train_vae(train_loader, valid_loader, input_shape, elbo_type):
    """ Training VAE with the specified image dataset
    :param train_loader: training image dataset loader
    :param valid_loader: validation image dataset loader
    :param input_shape: size of input image
    :param elbo_type: type of elbo function
    :return: trained model and training loss history
    """

    # load parameters
    epoch, lr, beta = train_dict["epoch"], train_dict["lr"], train_dict["beta"]

    # building VAE
    model = VariationalAutoencoder(input_shape)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Number of parameters: {num_params}")

    # specify elbo function
    if elbo_type == "binary":
        elbo_func = elbo_binary
    elif elbo_type == "l2":
        elbo_func = elbo_l2
    else:
        raise ValueError("Invalid ELBO type")

    # training loop
    model.train()
    train_loss, valid_loss = [], []
    for epoch in range(epoch):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training on epoch {epoch}...")
        train_ls, nbatch = 0., 0.

        for x_batch, _ in train_loader:
            batch_size = x_batch.shape[0]
            x_batch = x_batch.to(device)
            recon_batch, mu_batch, logvar_batch = model(x_batch)
            loss = elbo_func(x_batch, recon_batch, mu_batch, logvar_batch, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update loss and nbatch
            train_ls += loss.item() / batch_size
            nbatch += 1

        scheduler.step()

        # append training loss
        train_ls = train_ls / nbatch
        train_loss.append(train_ls)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish epoch {epoch} with loss {train_ls}")

        # append validation loss
        valid_ls = valid_vae(model, valid_loader, elbo_type)
        valid_loss.append(valid_ls)

    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)

    return model, train_loss, valid_loss


def valid_vae(model, valid_loader, elbo_type):
    """ Training VAE with the specified image dataset
    :param model: trained VAE model
    :param valid_loader: validation image dataset loader
    :param elbo_type: type of elbo function
    :return: validation loss
    """

    # load parameters
    beta = train_dict["beta"]

    # specify elbo function
    if elbo_type == "binary":
        elbo_func = elbo_binary
    elif elbo_type == "l2":
        elbo_func = elbo_l2
    else:
        raise ValueError("Invalid ELBO type")

    # set to evaluation mode
    model.eval()
    valid_loss, nbatch = 0., 0.
    for x_batch, _ in valid_loader:
        with torch.no_grad():
            batch_size = x_batch.shape[0]
            x_batch = x_batch.to(device)
            recon_batch, mu_batch, logvar_batch = model(x_batch)
            loss = elbo_func(x_batch, recon_batch, mu_batch, logvar_batch, beta)

            # update loss and nbatch
            valid_loss += loss.item() / batch_size
            nbatch += 1

    # report validation loss
    valid_loss = valid_loss / nbatch
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish validation with loss {valid_loss}")

    return valid_loss
