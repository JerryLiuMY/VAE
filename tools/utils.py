def to_img(x):
    """ This function takes as an input the
    :param x: reconstructed image
    :return:
    """

    x = x.clamp(0, 1)

    return x
