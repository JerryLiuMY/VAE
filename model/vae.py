from model.enc_dec import Encoder
from model.enc_dec import DecoderConv
from model.enc_dec import DecoderLinear
import torch.nn as nn
import torch


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape, vae_type):
        super(VariationalAutoencoder, self).__init__()

        if vae_type == "vae_conv":
            self.encoder = Encoder(input_shape)
            self.decoder = DecoderConv(input_shape)
        elif vae_type == "vae_linear":
            self.encoder = Encoder(input_shape)
            self.decoder = DecoderLinear(input_shape)

    def forward(self, x):
        # feed-forward function
        mu, logvar = self.encoder(x)
        latent = self.latent_sample(mu, logvar)
        x_rec = self.decoder(latent)

        return x_rec, mu, logvar

    def latent_sample(self, mu, logvar):
        # the re-parameterization trick
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
