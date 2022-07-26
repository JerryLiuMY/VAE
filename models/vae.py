import torch.nn.functional as F
from params.params import get_conv_size
from params.params import params_dict
import torch.nn as nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(input_size)
        self.decoder = Decoder(input_size)

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


class Block(nn.Module):
    def __init__(self, input_size):
        super(Block, self).__init__()
        self.channel = params_dict["channel"]
        self.kernel_size = params_dict["kernel_size"]
        self.stride = params_dict["stride"]
        self.padding = params_dict["padding"]
        self.dilation = params_dict["dilation"]
        self.hidden = params_dict["latent"]
        self.input_h, self.input_w = input_size


class Encoder(Block):
    def __init__(self, input_size):
        super(Encoder, self).__init__(input_size)

        # first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=self.kernel_size,
                               stride=self.stride, padding=self.padding)
        self.conv_h, self.conv_w = get_conv_size(self.input_h), get_conv_size(self.input_w)

        # second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel * 2, kernel_size=self.kernel_size,
                               stride=self.stride, padding=self.padding)
        self.conv_h, self.conv_w = get_conv_size(self.conv_h), get_conv_size(self.conv_w)

        # map to mu and variance
        self.fc_mu = nn.Linear(in_features=self.channel * 2 * self.conv_h * self.conv_w, out_features=self.hidden)
        self.fc_logvar = nn.Linear(in_features=self.channel * 2 * self.conv_h * self.conv_w, out_features=self.hidden)

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


class Decoder(Block):
    def __init__(self, input_size):
        super(Decoder, self).__init__(input_size)

        # linear layer
        self.fc = nn.Linear(in_features=self.latent, out_features=self.channel * 2 * self.conv_h * self.conv_w)

        # first convolutional layer
        self.conv2 = nn.ConvTranspose2d(in_channels=self.channel * 2, out_channels=self.channel,
                                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # second convolutional layer
        self.conv1 = nn.ConvTranspose2d(in_channels=self.channel, out_channels=self.channel,
                                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, x):
        # linear layer
        x = self.fc(x)

        # unflatten to channels
        x = x.view(x.size(0), self.c * 2, 7, 7)

        # convolution layers
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))

        return x
