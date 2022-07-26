import torch.nn.functional as F
from datetime import datetime
from params.params import get_conv_size
from params.params import params_dict
import torch.nn as nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_vae(input_data, params):
    """ Training VAE with image dataset
    :param input_data: input image dataset
    :param params: parameters
    :return:
    """

    # load parameters
    channel, latent_dim = params["channel"], params["latent_dim"]
    epoch, lr, beta = params["epoch"], params["lr"], params["beta"]

    # building VAE
    vae = VariationalAutoencoder(channel, latent_dim)
    vae = vae.to(device)
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Number of parameters: {num_params}")

    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)

    # set to training mode
    vae.train()
    for epoch in range(epoch):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training on epoch {epoch}")
        for img_batch, _ in input_data:
            img_batch = img_batch.to(device)
            ima_batch_recon, mu, logvar = vae(img_batch)
            loss = vae_loss(ima_batch_recon, img_batch, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.channel = params_dict["channel"]
        self.kernel_size = params_dict["kernel_size"]
        self.stride = params_dict["stride"]
        self.padding = params_dict["padding"]

        # first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=self.kernel_size,
                               stride=self.stride, padding=self.padding)
        self.conv_h, self.conv_w = get_conv_size(self.input_h), get_conv_size(self.input_w)

        # second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel*2, kernel_size=self.kernel_size,
                               stride=self.stride, padding=self.padding)
        self.conv_h, self.conv_w = get_conv_size(self.conv_h), get_conv_size(self.conv_w)

        # map to mu and variance
        self.fc_mu = nn.Linear(in_features=self.channel * 2 * self.conv_h * self.conv_w, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=self.channel * 2 * self.conv_h * self.conv_w, out_features=latent_dim)

    def forward(self, x):
        # convolution layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # flatten to vectors
        x = x.view(x.size(0), -1)

        # calculate mu & logvar
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        return x_mu, x_logvar


class Decoder(nn.Module, Encoder):
    def __init__(self, channels, latent_dim):
        super(Decoder, self).__init__()
        c = channels
        self.c = channels
        self.fc = nn.Linear(in_features=latent_dim, out_features=c * 2 * self.conv_h * self.conv_w)
        self.conv2 = nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # linear layer
        x = self.fc(x)

        # unflatten to channels
        x = x.view(x.size(0), self.c * 2, 7, 7)

        # convolution layers
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))

        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(capacity, latent_dims)
        self.decoder = Decoder(capacity, latent_dims)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)

        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        # the re-parameterization trick
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def vae_loss(recon_x, x, mu, logvar, beta):
    # reconstruction loss (dependent of image resolution)
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction="sum")

    # KL-divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_div
