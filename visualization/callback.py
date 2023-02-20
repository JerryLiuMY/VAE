from global_settings import OUTPUT_PATH
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
sns.set()


def plot_callback(elbo_type):
    """ Plot training and validation history
    :param elbo_type: type of elbo function
    :return: dataframe of z and x
    """

    # load training & validation loss
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    model_path = os.path.join(OUTPUT_PATH, f"model_{elbo_type}")

    train_loss = np.load(os.path.join(model_path, f"train_loss.npy"))
    valid_loss = np.load(os.path.join(model_path, f"valid_loss.npy"))

    # plot train_llh and valid_llh
    ax.set_title(f"Learning curve with elbo type {elbo_type}")
    ax.plot(train_loss, color=sns.color_palette()[0], label="train_llh")
    ax.plot(valid_loss, color=sns.color_palette()[1], label="valid_llh")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig
