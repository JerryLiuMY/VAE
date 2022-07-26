import torch.nn.functional as F
from params.params import get_conv_size
from params.params import params_dict
import torch.nn as nn
import torch


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(input_shape)
        self.decoder = Decoder(input_shape)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent = self.latent_sample(mu, logvar)
        x_recon = self.decoder(latent)

        return x_recon, mu, logvar

    def latent_sample(self, mu, logvar):
        # the re-parameterization trick
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


class Block(nn.Module):
    def __init__(self, input_shape):
        super(Block, self).__init__()
        self.channel = params_dict["channel"]
        self.kernel_size = params_dict["kernel_size"]
        self.stride = params_dict["stride"]
        self.padding = params_dict["padding"]
        self.dilation = params_dict["dilation"]
        self.hidden = params_dict["hidden"]
        self.input_h, self.input_w = input_shape


class Encoder(Block):
    def __init__(self, input_shape):
        super(Encoder, self).__init__(input_shape)

        # first convolutional layer
        self.deconv1 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=self.kernel_size,
                                 stride=self.stride, padding=self.padding)
        self.conv_h, self.conv_w = get_conv_size(self.input_h), get_conv_size(self.input_w)

        # second convolutional layer
        self.deconv2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel * 2, kernel_size=self.kernel_size,
                                 stride=self.stride, padding=self.padding)
        self.conv_h, self.conv_w = get_conv_size(self.conv_h), get_conv_size(self.conv_w)

        # map to mu and variance
        self.fc_mu = nn.Linear(in_features=self.channel * 2 * self.conv_h * self.conv_w, out_features=self.hidden)
        self.fc_logvar = nn.Linear(in_features=self.channel * 2 * self.conv_h * self.conv_w, out_features=self.hidden)

    def forward(self, x):
        # convolution layers
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))

        # flatten to vectors
        x = x.view(x.size(0), -1)

        # calculate mu & logvar
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        return x_mu, x_logvar


class Decoder(Encoder, Block):
    def __init__(self, input_shape):
        super(Decoder, self).__init__(input_shape)

        # linear layer
        self.fc = nn.Linear(in_features=self.hidden, out_features=self.channel * 2 * self.conv_h * self.conv_w)

        # first convolutional layer
        self.deconv2 = nn.ConvTranspose2d(in_channels=self.channel * 2, out_channels=self.channel,
                                          kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # second convolutional layer
        self.deconv1 = nn.ConvTranspose2d(in_channels=self.channel, out_channels=1,
                                          kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, x):
        # linear layer
        x = self.fc(x)

        # unflatten to channels
        x = x.view(x.size(0), self.channel * 2, self.conv_h, self.conv_w)

        # convolution layers
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv1(x))

        return x
