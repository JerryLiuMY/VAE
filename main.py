from loader.loader import load_data
from models.train import train_vae


def experiment(dataset):
    data_loader, input_size = load_data(dataset)
    train_vae(data_loader, input_size)


if __name__ == "__main__":
    experiment("mnist")
