from torchvision import datasets
from torchvision.transforms import ToTensor

# Functions to import and classify the MNIST dataset of handwritten images


def trainingDataset():
    return datasets.MNIST(root="data", train=True, transform=ToTensor(), download=True)


def testingDataset():
    return datasets.MNIST(root="data", train=False, transform=ToTensor(), download=True)
