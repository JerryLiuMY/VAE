from loader.loader import load_data
from models.train import train_vae
from models.train import valid_vae
from global_settings import OUTPUT_PATH
import numpy as np
import torch
import os


def experiment(dataset):
    """ Perform experiment on the dataset
    :param dataset:
    :return:
    """

    # load data and perform training
    train_loader, valid_loader, input_shape = load_data(dataset)
    model, train_loss = train_vae(train_loader, input_shape)
    valid_loss = valid_vae(model, valid_loader)

    # save model and loss
    model_path = os.path.join(OUTPUT_PATH, "model")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    torch.save(model, os.path.join(model_path, "model.pth"))
    np.save(os.path.join(model_path, "train_loss.npy"), train_loss)
    np.save(os.path.join(model_path, "valid_loss.npy"), valid_loss)


if __name__ == "__main__":
    experiment("mnist")
