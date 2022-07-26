import torchvision.transforms as transforms
import torchvision.datasets as datasets
from global_settings import DATA_PATH
from torch.utils.data import DataLoader
from params.params import params_dict


def load_mnist():
    batch_size = params_dict["batch_size"]
    img_transform = transforms.Compose([transforms.ToTensor()])
    data = datasets.MNIST(DATA_PATH, download=True, train=True, transform=img_transform)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return data_loader
