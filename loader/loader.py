import torchvision.transforms as transforms
import torchvision.datasets as datasets
from global_settings import DATA_PATH
from torch.utils.data import DataLoader
from params.params import batch_size


def load_data(dataset_name):
    if dataset_name == "mnist":
        load_func = datasets.MNIST
    else:
        raise ValueError("Invalid dataset name")

    img_transform = transforms.Compose([transforms.ToTensor()])
    data = load_func(DATA_PATH, download=True, train=True, transform=img_transform)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    input_size = (data[0][0].shape[1], data[0][0].shape[2])

    return data_loader, input_size
