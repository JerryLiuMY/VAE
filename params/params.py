import math
batch_size = 128
params_dict = {"channel": 64, "kernel_size": 4, "stride": 2, "padding": 1, "dilation": 1, "hidden": 2}
train_dict = {"epoch": 50, "lr": 0.001, "beta": 1}


def get_conv_size(input_size):
    """ calculate convolution output size from input size
    :param input_size: input size of convolutional layer
    :return: output size from convolutional layer
    """

    num = input_size + 2 * params_dict["padding"] - params_dict["dilation"] * (params_dict["kernel_size"] - 1) - 1
    den = params_dict["stride"]
    conv_size = math.floor(num/den + 1)

    return conv_size
