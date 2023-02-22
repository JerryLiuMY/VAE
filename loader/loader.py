import torchvision.transforms as transforms
import torchvision.datasets as datasets
from global_settings import DATA_PATH
from torch.utils.data import DataLoader
from params.params import batch_size
from datetime import datetime


def load_data(dataset):
    """ Load data from the specified dataset as data loader
    :param dataset: dataset name
    :return: dataset loader and input shape
    """

    if dataset == "mnist":
        load_func = datasets.MNIST
    else:
        raise ValueError("Invalid dataset name")

    img_transform = transforms.Compose([transforms.ToTensor()])
    train_data = load_func(DATA_PATH, download=True, train=True, transform=img_transform)
    valid_data = load_func(DATA_PATH, download=True, train=False, transform=img_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    input_size = (train_data[0][0].shape[1], train_data[0][0].shape[2])
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Loaded {dataset.upper()} data with input shape {input_size}")

    return train_loader, valid_loader, input_size


def sort_digits(data_loader):
    """ Group tensor by digits label
    :param data_loader: data loader
    :return: list of image tensors grouped by digits label
    """

    digits = [[] for _ in range(10)]
    for valid_batch, label_batch in data_loader:
        for i in range(valid_batch.size(0)):
            digits[label_batch[i]].append(valid_batch[i:i + 1])
        if sum(len(d) for d in digits) >= 1000:
            break

    return digits
