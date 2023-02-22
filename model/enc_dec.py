from params.params import get_conv_size, params_dict
from torch.nn import functional as F
from torch import nn as nn
import torch


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
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class DecoderConv(Encoder, Block):
    def __init__(self, input_shape):
        # Kingma & Welling -- Two conv layers
        super(DecoderConv, self).__init__(input_shape)

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


class DecoderLinear(Encoder, Block):
    def __init__(self, input_shape):
        # Kingma & Welling -- Two linear layers
        super(DecoderLinear, self).__init__(input_shape)

        # linear layer
        self.fc = nn.Linear(
            in_features=self.hidden,
            out_features=self.channel * 2 * self.conv_h * self.conv_w
        )

        # first linear layer
        self.delinear2 = nn.Linear(
            in_features=self.channel * 2 * self.conv_h * self.conv_w,
            out_features=self.channel * self.conv_h * self.conv_w
        )

        # second linear layer
        self.delinear1 = nn.Linear(
            in_features=self.channel * self.conv_h * self.conv_w,
            out_features=self.conv_h * self.conv_w
        )

    def forward(self, x):
        # linear layer
        x = self.fc(x)

        # convolution layers
        x = F.relu(self.delinear2(x))
        x = torch.sigmoid(self.delinear1(x))

        # unflatten to channels
        x = x.view(x.size(0), 1, self.conv_h, self.conv_w)

        return x
