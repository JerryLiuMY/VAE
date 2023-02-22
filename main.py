from loader.loader import load_data
from learning.train import train_vae
from global_settings import OUTPUT_PATH
from visualization.recon import plot_recon
import numpy as np
import torch
import os


def experiment(dataset, model_type, elbo_type, hidden):
    """ Perform experiment on the dataset
    :param dataset: dataset name
    :param model_type: type of model to use
    :param elbo_type: type of loss function
    :param hidden: hidden dimension of the latent space
    """

    # define paths
    model_path = os.path.join(OUTPUT_PATH, f"{model_type}-{elbo_type}")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # load data and perform training
    train_loader, valid_loader, input_shape = load_data(dataset)
    model, train_loss, valid_loss = train_vae(train_loader, valid_loader, input_shape, model_type, elbo_type, hidden)

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
    image_set, labels = next(iter(valid_loader))

    # plot visualizations
    visual_path = os.path.join(OUTPUT_PATH, "visualizations")
    if not os.path.isdir(visual_path):
        os.mkdir(visual_path)

    plot_recon(model, image_set)


if __name__ == "__main__":
    experiment(dataset="mnist", model_type="vae_conv", elbo_type="binary", hidden=3)
