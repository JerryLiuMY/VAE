from loader.loader import load_data
from learning.train import train_vae
from global_settings import OUTPUT_PATH
from visualization.recon import plot_recon
from visualization.inter import plot_inter
from loader.loader import sort_digits
import numpy as np
import torch
import os


def experiment(dataset, elbo_type):
    """ Perform experiment on the dataset
    :param dataset: dataset name
    :param elbo_type: type of loss function
    """

    # define paths
    model_path = os.path.join(OUTPUT_PATH, f"model_{elbo_type}")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # load data and perform training
    train_loader, valid_loader, input_shape = load_data(dataset)
    model, train_loss, valid_loss = train_vae(train_loader, valid_loader, input_shape, elbo_type)

    # save model and loss
    torch.save(model, os.path.join(model_path, f"model.pth"))
    np.save(os.path.join(model_path, f"train_loss.npy"), train_loss)
    np.save(os.path.join(model_path, f"valid_loss.npy"), valid_loss)


def visualize(dataset):
    """ Perform various visualizations
    :param dataset: dataset name
    """

    # load data and model
    model_path = os.path.join(OUTPUT_PATH, "model")
    model = torch.load(os.path.join(model_path, "model.pth"), map_location=torch.device("cpu"))
    train_loader, valid_loader, input_shape = load_data(dataset)
    digit_set = sort_digits(valid_loader)
    image_set, labels = next(iter(valid_loader))

    # plot visualizations
    visual_path = os.path.join(OUTPUT_PATH, "visualizations")
    if not os.path.isdir(visual_path):
        os.mkdir(visual_path)

    plot_recon(model, image_set)
    plot_inter(model, digit_set, d1=3, d2=9)


if __name__ == "__main__":
    experiment("mnist", "l2")
